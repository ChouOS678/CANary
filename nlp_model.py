"""CAN 序列化 NLP 训练与预测。

迁移目标：
- 不再使用 Transformer 主干
- 改为 CPU 友好的经典 NLP 分类器
- 保留“CAN 序列 -> token/text -> 分类”的课程设计叙事

默认提供两种模型：
- LogisticRegression
- LinearSVC
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from config import LABELS
from utils import logger

from nlp_data import CanTextVectorizer, TokenizationConfig, generate_can_sequences, prepare_nlp_dataset


@dataclass
class NLPModelBundle:
    name: str
    vectorizer: CanTextVectorizer
    classifier: Any

    def predict(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        return self.classifier.predict(x)


def _build_classifier(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000, n_jobs=None)
    if model_name == "linearsvc":
        return LinearSVC()
    raise ValueError(f"Unknown NLP model: {model_name}")


def train_nlp_model(
    nlp_dataset: dict[str, Any],
    group_config: dict[str, object],
) -> tuple[NLPModelBundle, dict[str, Any]]:
    texts: list[str] = nlp_dataset["texts"]
    labels: np.ndarray = nlp_dataset["labels"]
    vectorizer: CanTextVectorizer = nlp_dataset["vectorizer"]

    model_name = str(group_config.get("nlp_model", "logreg")).lower()
    classifier = _build_classifier(model_name)

    x = nlp_dataset["features"]
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        labels,
        test_size=0.22,
        random_state=42,
        stratify=labels,
    )

    start = time.perf_counter()
    classifier.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    val_pred = classifier.predict(x_val)
    predict_time = time.perf_counter() - start

    accuracy = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average="macro")

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "train_time_sec": round(train_time, 4),
        "predict_time_sec": round(predict_time, 4),
        "vocab_size": int(nlp_dataset["vocab_size"]),
        "seq_len": int(nlp_dataset["seq_len"]),
        "model_name": model_name,
        "n_samples": int(nlp_dataset["n_samples"]),
        "gpu_available": bool(nlp_dataset.get("gpu_available", False)),
        "n_features": int(x.shape[1]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores: list[float] = []
    for train_idx, test_idx in skf.split(x, labels):
        clf = _build_classifier(model_name)
        clf.fit(x[train_idx], labels[train_idx])
        cv_pred = clf.predict(x[test_idx])
        cv_scores.append(float(accuracy_score(labels[test_idx], cv_pred)))
    metrics["cv_mean_accuracy"] = float(np.mean(cv_scores))
    metrics["cv_std_accuracy"] = float(np.std(cv_scores))
    metrics["cv_fold_accuracies"] = cv_scores
    metrics["cv_folds"] = len(cv_scores)

    bundle = NLPModelBundle(name=model_name, vectorizer=vectorizer, classifier=classifier)
    logger.info(
        f"[NLP] model={model_name} acc={accuracy:.4f} f1={f1:.4f} "
        f"train={train_time:.2f}s predict={predict_time:.2f}s"
    )
    return bundle, metrics


def predict_nlp_sequences(model: NLPModelBundle, nlp_dataset: dict[str, Any]) -> list[dict[str, Any]]:
    texts: list[str] = nlp_dataset["texts"]
    x = model.vectorizer.transform(texts)
    preds = model.classifier.predict(x)
    results: list[dict[str, Any]] = []
    for idx, pred in enumerate(preds):
        results.append(
            {
                "predicted_label": LABELS[int(pred)],
                "predicted_label_idx": int(pred),
                "confidence": 1.0,
                "sequence_text": texts[idx],
            }
        )
    return results
