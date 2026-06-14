"""CAN 序列化 NLP 数据层。

这个模块把 CAN 消息视作领域文本：
- 一条消息映射为一个 token
- 一段窗口内的消息序列映射为一个“句子”
- 使用稀疏文本表示供 CPU 侧传统 NLP 分类器训练与预测

保留 GPU 预留接口：如果环境中有 CuPy / GPU，可在批量编码时加速；
若没有，则自动回退到 NumPy 实现，保证工程可运行。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import LABELS

try:
    import cupy as cp  # type: ignore

    GPU_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    GPU_AVAILABLE = False


@dataclass(frozen=True)
class TokenizationConfig:
    max_seq_len: int = 64
    use_tfidf: bool = True
    ngram_range: tuple[int, int] = (1, 2)


def _message_to_token(msg: dict[str, Any]) -> str:
    can_id = int(msg.get("can_id", 0))
    dlc = int(msg.get("dlc", 0))
    b0 = int(msg.get("data_byte0", 0))
    b1 = int(msg.get("data_byte1", 0))
    interval = float(msg.get("interval_ms", 0.0))
    interval_bin = min(9, int(interval // 10))
    payload_sig = f"{b0:02X}{b1:02X}"
    return f"ID{can_id:03X}_D{dlc}_P{payload_sig}_T{interval_bin}"


def _window_to_text(sequence: list[dict[str, Any]], max_seq_len: int) -> str:
    tokens = [_message_to_token(msg) for msg in sequence[:max_seq_len]]
    return " ".join(tokens)


def generate_can_text_samples(
    group_config: dict[str, object],
    seq_len: int = 64,
) -> list[dict[str, Any]]:
    """生成 CAN 文本样本。"""
    from nlp_data import generate_can_sequences  # 复用序列生成逻辑

    samples = generate_can_sequences(group_config, seq_len=seq_len)
    text_samples: list[dict[str, Any]] = []
    for item in samples:
        text_samples.append(
            {
                "text": _window_to_text(item["sequence"], seq_len),
                "label": item["label"],
                "label_idx": item["label_idx"],
                "sequence": item["sequence"],
            }
        )
    return text_samples


# ---------------------------------------------------------------------------
# 原始 CAN 序列生成逻辑
# ---------------------------------------------------------------------------

ECU_IDS = {
    "engine": list(range(0x100, 0x110)),
    "trans": list(range(0x200, 0x210)),
    "abs": list(range(0x300, 0x310)),
    "body": list(range(0x400, 0x410)),
    "steering": list(range(0x500, 0x508)),
    "infotain": list(range(0x600, 0x608)),
    "diag": list(range(0x700, 0x710)),
    "gateway": list(range(0x750, 0x758)),
}

ECU_PERIODS_MS = {
    "engine": 10,
    "trans": 20,
    "abs": 20,
    "body": 50,
    "steering": 20,
    "infotain": 100,
    "diag": 500,
    "gateway": 100,
}


def _generate_normal_sequence(seq_len: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Vectorized CAN message sequence generation via NumPy batch ops.

    Pre-generates all random values at once to avoid per-message Python
    call overhead, and uses ``np.argmin`` over a flat array instead of
    ``min(dict.items(), key=...)`` for ECU arbitration.
    """
    ecu_names = list(ECU_IDS.keys())
    n_ecu = len(ecu_names)
    ecu_periods = np.array([ECU_PERIODS_MS[n] for n in ecu_names], dtype=np.float64)

    # ── batch-generate all random values at once ──
    pool = rng.uniform(0.0, 1.0, seq_len * 12)
    pi = 0  # pool index

    def _u() -> float:
        nonlocal pi
        v = float(pool[pi])
        pi += 1
        return v

    next_send = np.array([_u() * ECU_PERIODS_MS[n] for n in ecu_names])
    sensor = np.array([[_u() * 255.0, _u() * 255.0, _u() * 255.0]
                       for _ in range(n_ecu)])

    seq: list[dict[str, Any]] = []
    for _ in range(seq_len):
        ecu_idx = int(np.argmin(next_send))
        send_time = next_send[ecu_idx]
        period = float(ecu_periods[ecu_idx])

        # select CAN ID for this ECU
        ids = ECU_IDS[ecu_names[ecu_idx]]
        can_id = int(ids[int(_u() * len(ids))])

        # sensor drift with jitter
        delta = np.array([(_u() - 0.5) * 24.0,
                          (_u() - 0.5) * 24.0,
                          (_u() - 0.5) * 24.0])
        data = np.clip(sensor[ecu_idx] + delta, 0.0, 255.0)
        sensor[ecu_idx] = data

        interval = period + (_u() - 0.5) * period * 0.2

        seq.append({
            "can_id": can_id,
            "dlc": 8,
            "data_byte0": int(data[0]),
            "data_byte1": int(data[1]),
            "data_byte2": int(data[2]),
            "interval_ms": float(interval),
        })

        # schedule next transmission
        next_send[ecu_idx] = send_time + period + (_u() - 0.5) * period * 0.1

    return seq


def generate_can_sequences(group_config: dict[str, object], seq_len: int = 64) -> list[dict[str, Any]]:
    samples_per_label = int(group_config["samples_per_label"])
    seed = int(group_config["training_seed"])
    rng = np.random.default_rng(seed)
    all_samples: list[dict[str, Any]] = []
    for label_idx, label in enumerate(LABELS):
        for _ in range(samples_per_label):
            seq = _generate_normal_sequence(seq_len, rng)
            all_samples.append({"sequence": seq, "label": label, "label_idx": label_idx})
    return all_samples


# ---------------------------------------------------------------------------
# NLP 表示层
# ---------------------------------------------------------------------------

class CanTextVectorizer:
    def __init__(self, config: TokenizationConfig | None = None):
        self.config = config or TokenizationConfig()
        self.vectorizer = (
            TfidfVectorizer(ngram_range=self.config.ngram_range, lowercase=False)
            if self.config.use_tfidf
            else CountVectorizer(ngram_range=self.config.ngram_range, lowercase=False)
        )
        self.vocab_size_: int = 0
        self.gpu_available = GPU_AVAILABLE

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self.vocab_size_ = len(self.vectorizer.vocabulary_)
        return self

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        mat = self.vectorizer.fit_transform(texts)
        self.vocab_size_ = len(self.vectorizer.vocabulary_)
        return mat

    def get_feature_names(self) -> list[str]:
        return list(self.vectorizer.get_feature_names_out())


def prepare_nlp_dataset(samples: list[dict[str, Any]], seq_len: int = 64) -> dict[str, Any]:
    texts = [_window_to_text(sample["sequence"], seq_len) for sample in samples]
    labels = np.asarray([sample["label_idx"] for sample in samples], dtype=np.int64)
    vectorizer = CanTextVectorizer(TokenizationConfig(max_seq_len=seq_len))
    x = vectorizer.fit_transform(texts)
    return {
        "texts": texts,
        "labels": labels,
        "features": x,
        "vectorizer": vectorizer,
        "vocab_size": vectorizer.vocab_size_,
        "seq_len": seq_len,
        "label_names": LABELS,
        "n_samples": len(samples),
        "gpu_available": GPU_AVAILABLE,
    }
