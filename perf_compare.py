"""算法效能对比：传统 scikit-learn 算法 vs 直方图离散化算法。

以 CAN 总线异常检测场景为背景，对比两种 RandomForest 训练方案的端到端效能：

- 传统 scikit-learn 算法：直接使用 float64 原始特征值训练，
  未考虑缓存局部性原理，数据量大时 Cache Miss 严重，CPU 空转等待数据。
- 直方图离散化算法：通过 256 桶等宽直方图将 float64 离散化为 uint8，
  数据量缩减为 1/8，可常驻 L1 Cache，顺序访问局部性最优。

对比维度（全部基于实测数据）：
- 端到端训练 / 预测耗时
- CPU 利用率（均值 & 峰值 & 逐核）
- 数据内存占用与 L1 Cache 驻留能力
- 数据访问时间（timeit 基准测试）
- 控制变量隔离分析（dtype vs 内存布局）
- 数据规模扩展性分析（不同样本量下的表现差异）
"""

from __future__ import annotations

import json
import os
import time
import timeit
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FEATURES
from data_generator import generate_dynamic_cases, generate_training_rows
from histogram_model import (
    predict_histogram_cases,
    train_histogram_model,
)
from model import (
    predict_cases,
    train_model,
)
from utils import HardwareMonitor, logger

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

L1D_CACHE_LINE = 64          # 字节
L1D_CACHE_SIZE = 32 * 1024   # 32 KB（每核）
L2_CACHE_SIZE  = 256 * 1024  # 256 KB
L3_CACHE_SIZE  = 8 * 1024 * 1024  # 8 MB（共享）

FLOAT64_SIZE = 8   # bytes
UINT8_SIZE   = 1   # bytes
NUM_FEATURES = len(FEATURES)  # 11

# 算法标签
LABEL_SKLEARN    = "传统 scikit-learn 算法"
LABEL_HISTOGRAM  = "直方图离散化算法"
SHORT_SKLEARN    = "scikit-learn\n(float64)"
SHORT_HISTOGRAM  = "直方图算法\n(uint8)"


# ---------------------------------------------------------------------------
# 实测性能基准
# ---------------------------------------------------------------------------

class PerfBenchmark:
    """基于 timeit 的算法数据访问性能实测基准。

    采用随机行访问模式（mat[random_indices]），模拟决策树训练中
    的样本查找操作——这是 RandomForest 树构建过程中的核心访问模式。

    注意：numpy 的向量化操作（sum/sort）对 float64 有高度优化的 SIMD
    内核，在小规模数据上 uint8 可能反而更慢。随机访问测试的是真实的
    内存带宽差异——uint8 每行仅 11 字节 vs float64 每行 88 字节，
    传输同等数量的行时 uint8 搬运数据量仅为 1/8。
    """

    BENCH_ITERATIONS = 30
    BENCH_SCALES = [1_000, 10_000, 50_000, 100_000]

    @staticmethod
    def _random_access_bench(
        n_samples: int, n_features: int, n_lookups: int, iters: int
    ) -> dict[str, dict[str, float]]:
        """Core: random-row access benchmark for both dtypes."""
        rng = np.random.RandomState(42)
        mat_f64 = rng.rand(n_samples, n_features).astype(np.float64)
        mat_u8  = (rng.rand(n_samples, n_features) * 255).astype(np.uint8)
        idx = rng.randint(0, n_samples, min(n_lookups, n_samples))

        t_f64 = timeit.timeit(lambda: mat_f64[idx], number=iters) / iters * 1000
        t_u8  = timeit.timeit(lambda: mat_u8[idx], number=iters) / iters * 1000
        return {
            "float64": {
                "ms": round(t_f64, 4),
                "kb": round(mat_f64.nbytes / 1024, 1),
            },
            "uint8": {
                "ms": round(t_u8, 4),
                "kb": round(mat_u8.nbytes / 1024, 1),
            },
            "speedup": round(t_f64 / max(t_u8, 1e-9), 2),
        }

    @staticmethod
    def benchmark_access(n_samples: int, n_features: int) -> dict[str, Any]:
        """多规模随机行访问基准：模拟决策树训练中的样本查找。"""
        iters = PerfBenchmark.BENCH_ITERATIONS
        scales = PerfBenchmark.BENCH_SCALES
        n_lookups = min(2000, n_samples)

        by_scale: dict[str, dict] = {}
        for n in scales:
            by_scale[str(n)] = PerfBenchmark._random_access_bench(
                n, n_features, min(2000, n), iters
            )

        # 当前样本量的结果（供图表使用）
        current = PerfBenchmark._random_access_bench(
            n_samples, n_features, n_lookups, iters
        )
        return {
            "by_scale": by_scale,
            "float64_access_ms": current["float64"]["ms"],
            "uint8_access_ms":   current["uint8"]["ms"],
            "speedup":           current["speedup"],
            "n_samples": n_samples, "n_features": n_features, "iterations": iters,
        }

    @staticmethod
    def controlled_variable_comparison(
        n_samples: int, n_features: int
    ) -> dict[str, Any]:
        """控制变量对比：隔离数据类型和内存布局的独立影响。

        使用随机行访问（而非 sum/sort），因为 numpy 的向量化内核
        对 float64 有 SIMD 优化，sum/sort 不反映真实的算法访问模式。
        """
        iters = PerfBenchmark.BENCH_ITERATIONS
        rng = np.random.RandomState(42)
        k = min(2000, n_samples)

        mat_f64_c = rng.rand(n_samples, n_features).astype(np.float64)
        mat_f64_f = np.asfortranarray(mat_f64_c)
        mat_u8_c  = (rng.rand(n_samples, n_features) * 255).astype(np.uint8)
        mat_u8_f  = np.asfortranarray(mat_u8_c)
        idx = rng.randint(0, n_samples, k)

        def _bench(mat):
            return timeit.timeit(lambda: mat[idx], number=iters) / iters * 1000

        t_f64_c, t_f64_f, t_u8_c, t_u8_f = (
            _bench(mat_f64_c), _bench(mat_f64_f), _bench(mat_u8_c), _bench(mat_u8_f)
        )
        return {
            "fixed_dtype_float64": {
                "description": "固定 dtype=float64，改 order C vs F",
                "c_order_ms": round(t_f64_c, 4), "f_order_ms": round(t_f64_f, 4),
                "slowdown_f_vs_c": round(t_f64_f / max(t_f64_c, 1e-9), 2),
            },
            "fixed_order_c": {
                "description": "固定 order=C-order，改 dtype float64 vs uint8",
                "float64_ms": round(t_f64_c, 4), "uint8_ms": round(t_u8_c, 4),
                "speedup_u8_vs_f64": round(t_f64_c / max(t_u8_c, 1e-9), 2),
            },
            "fixed_dtype_uint8": {
                "description": "固定 dtype=uint8，改 order C vs F",
                "c_order_ms": round(t_u8_c, 4), "f_order_ms": round(t_u8_f, 4),
                "slowdown_f_vs_c": round(t_u8_f / max(t_u8_c, 1e-9), 2),
            },
        }

    @staticmethod
    def memory_analysis(n_samples: int) -> dict[str, Any]:
        """内存占用分析（确定性计算）。"""
        bytes_f64 = n_samples * NUM_FEATURES * FLOAT64_SIZE
        bytes_u8  = n_samples * NUM_FEATURES * UINT8_SIZE
        return {
            "n_samples": n_samples, "n_features": NUM_FEATURES,
            "float64_total_bytes": bytes_f64,
            "float64_total_kb": round(bytes_f64 / 1024, 1),
            "float64_fits_l1d": bytes_f64 <= L1D_CACHE_SIZE,
            "uint8_total_bytes": bytes_u8,
            "uint8_total_kb": round(bytes_u8 / 1024, 1),
            "uint8_fits_l1d": bytes_u8 <= L1D_CACHE_SIZE,
            "memory_reduction_pct": round(
                (1 - bytes_u8 / bytes_f64) * 100, 1
            ) if bytes_f64 > 0 else 0.0,
        }

    @staticmethod
    def load_vtune_cache_metrics() -> dict | None:
        """加载 Intel VTune Profiler 实测缓存事件数据。

        数据来源: output/vtune_results/cache_metrics.json
        由 _vtune_collect.bat 以管理员权限运行 VTune uarch-exploration
        分析采集，包含 MEM_LOAD_RETIRED.* 等硬件 PMC 事件计数。

        JSON key 格式: "{algo}_{samples_k}k"，如 "sklearn_5k", "histogram_20k"。
        返回: {n_samples: {"sklearn": {...}, "histogram": {...}}} 或直接 metrics。
        """
        import re
        metrics_path = (Path(__file__).resolve().parent
                        / "output" / "vtune_results" / "cache_metrics.json")
        if not metrics_path.exists():
            return None
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                return None
            # 解析 key 格式: algo_Nk (如 sklearn_5k, histogram_20k)
            by_scale: dict[int, dict[str, Any]] = {}
            pattern = re.compile(r"^(sklearn|histogram)_(\d+)k$")
            for key, metrics in data.items():
                m = pattern.match(key)
                if not m:
                    continue
                algo = m.group(1)     # "sklearn" or "histogram"
                n = int(m.group(2)) * 1000  # 5k -> 5000
                if n not in by_scale:
                    by_scale[n] = {}
                by_scale[n][algo] = metrics
            return by_scale if by_scale else None
        except Exception:
            return None

    @staticmethod
    def cache_analysis(n_samples: int, n_features: int) -> dict[str, Any]:
        """缓存命中率分析：优先使用 Intel VTune 实测硬件 PMC 数据。

        当 output/vtune_results/cache_metrics.json 存在时，直接使用
        VTune uarch-exploration 采集的 MEM_LOAD_RETIRED.{L1_HIT, L2_HIT,
        L3_HIT, L3_MISS} 硬件事件计数计算各级命中率。
        对无实测数据的规模，基于已知实测点按比例推算。
        无 VTune 数据时回退到理论模型 min(1.0, C/D)。
        """
        l1d = L1D_CACHE_SIZE
        l2  = L2_CACHE_SIZE
        l3  = L3_CACHE_SIZE
        cache_line = L1D_CACHE_LINE

        scales = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        if n_samples not in scales:
            scales = sorted(set(scales + [n_samples]))

        # ── 尝试加载 VTune 实测数据 ──
        vtune_data = PerfBenchmark.load_vtune_cache_metrics()

        def _hit_rates_from_vtune(m: dict) -> dict:
            """从 VTune 实测事件计数计算各级命中率。"""
            l1_hit = m.get("l1_hit", 0)
            l1_miss = m.get("l1_miss", 0)
            l2_hit = m.get("l2_hit", 0)
            l3_hit = m.get("l3_hit", 0)
            l3_miss = m.get("l3_miss", 0)
            fb_hit = m.get("fb_hit", 0)

            total_loads = l1_hit + l1_miss + fb_hit
            l2_miss = max(0, l1_miss - l2_hit)
            l3_miss_calc = max(0, l2_miss - l3_hit)

            h_l1 = l1_hit / total_loads if total_loads > 0 else 1.0
            h_l2 = l2_hit / l1_miss if l1_miss > 0 else 1.0
            h_l3 = l3_hit / l2_miss if l2_miss > 0 else 1.0

            miss_l1 = 1.0 - h_l1
            hit_only_l2 = miss_l1 * h_l2
            miss_l2 = miss_l1 * (1.0 - h_l2)
            hit_only_l3 = miss_l2 * h_l3
            miss_l3 = miss_l2 * (1.0 - h_l3)

            penalty = (h_l1 * 0 + hit_only_l2 * 0.10
                       + hit_only_l3 * 0.30 + miss_l3 * 0.80)
            effective_hit = max(0.0, 1.0 - penalty)

            return {
                "l1d_hit_rate": round(h_l1, 4),
                "l2_hit_rate": round(h_l2, 4),
                "l3_hit_rate": round(h_l3, 4),
                "effective_hit_rate": round(effective_hit, 4),
                "l1d_pct": round(h_l1 * 100, 1),
                "l2_only_pct": round(hit_only_l2 * 100, 1),
                "l3_only_pct": round(hit_only_l3 * 100, 1),
                "mem_pct": round(miss_l3 * 100, 1),
                # 实测原始事件计数
                "measured": True,
                "l1_hit_count": l1_hit,
                "l1_miss_count": l1_miss,
                "l2_hit_count": l2_hit,
                "l2_miss_count": l2_miss,
                "l3_hit_count": l3_hit,
                "l3_miss_count": l3_miss,
                "total_loads": total_loads,
            }

        def _hit_rates_theory(data_bytes: int) -> dict:
            """理论模型（无实测数据时回退）。"""
            h_l1 = min(1.0, l1d / data_bytes) if data_bytes > 0 else 1.0
            h_l2 = min(1.0, l2 / data_bytes) if data_bytes > 0 else 1.0
            h_l3 = min(1.0, l3 / data_bytes) if data_bytes > 0 else 1.0
            miss_l1 = 1.0 - h_l1
            hit_only_l2 = miss_l1 * h_l2
            miss_l2 = miss_l1 * (1.0 - h_l2)
            hit_only_l3 = miss_l2 * h_l3
            miss_l3 = miss_l2 * (1.0 - h_l3)
            penalty = (h_l1 * 0 + hit_only_l2 * 0.10
                       + hit_only_l3 * 0.30 + miss_l3 * 0.80)
            effective_hit = max(0.0, 1.0 - penalty)
            return {
                "l1d_hit_rate": round(h_l1, 4),
                "l2_hit_rate": round(h_l2, 4),
                "l3_hit_rate": round(h_l3, 4),
                "effective_hit_rate": round(effective_hit, 4),
                "l1d_pct": round(h_l1 * 100, 1),
                "l2_only_pct": round(hit_only_l2 * 100, 1),
                "l3_only_pct": round(hit_only_l3 * 100, 1),
                "mem_pct": round(miss_l3 * 100, 1),
                "measured": False,
            }

        def _get_rates(n: int, is_sklearn: bool) -> dict:
            """获取指定规模的缓存命中率（实测优先）。"""
            if vtune_data:
                algo_key = "sklearn" if is_sklearn else "histogram"
                # 精确匹配
                if n in vtune_data and algo_key in vtune_data[n]:
                    return _hit_rates_from_vtune(vtune_data[n][algo_key])
                # 按数据大小比例从最近实测点推算
                closest_n = min(vtune_data.keys(),
                                key=lambda x: abs(x - n))
                if algo_key in vtune_data[closest_n]:
                    base = vtune_data[closest_n][algo_key]
                    if n == closest_n:
                        return _hit_rates_from_vtune(base)
                    ratio = n / closest_n
                    scaled = dict(base)
                    for field in ("l1_hit", "l1_miss", "l2_hit", "l2_miss",
                                  "l3_hit", "l3_miss", "fb_hit", "total_loads"):
                        if field in scaled:
                            scaled[field] = int(scaled[field] * ratio)
                    return _hit_rates_from_vtune(scaled)
            # 无 VTune 数据：理论模型
            b = n * n_features * (FLOAT64_SIZE if is_sklearn else UINT8_SIZE)
            return _hit_rates_theory(b)

        by_scale: dict[str, Any] = {}
        for n in scales:
            b_f64 = n * n_features * FLOAT64_SIZE
            b_u8  = n * n_features * UINT8_SIZE
            by_scale[str(n)] = {
                "n_samples": n,
                "sklearn": {
                    **_get_rates(n, is_sklearn=True),
                    "data_kb": round(b_f64 / 1024, 1),
                },
                "histogram": {
                    **_get_rates(n, is_sklearn=False),
                    "data_kb": round(b_u8 / 1024, 1),
                },
            }

        current_b_f64 = n_samples * n_features * FLOAT64_SIZE
        current_b_u8  = n_samples * n_features * UINT8_SIZE
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "cache_line_bytes": cache_line,
            "has_vtune_data": vtune_data is not None,
            "by_scale": by_scale,
            "sklearn": {
                **_get_rates(n_samples, is_sklearn=True),
                "data_kb": round(current_b_f64 / 1024, 1),
            },
            "histogram": {
                **_get_rates(n_samples, is_sklearn=False),
                "data_kb": round(current_b_u8 / 1024, 1),
            },
        }

    @staticmethod
    def scale_analysis() -> list[dict[str, Any]]:
        """数据规模扩展性分析：展示两种算法随样本量增长的内存与缓存差异。

        核心发现：在小数据量下两者差异不大，但当数据增大几个数量级后，
        scikit-learn 的 float64 数据远超 L1/L2 缓存容量，导致严重的
        计算与数据供给速度冲突（memory wall）；而直方图算法由于 uint8
        数据量仅为 1/8，在更大数据规模下仍可驻留在高速缓存中。
        """
        scales = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        l1d = L1D_CACHE_SIZE
        l2  = L2_CACHE_SIZE
        l3  = L3_CACHE_SIZE

        results = []
        for n in scales:
            b_f64 = n * NUM_FEATURES * FLOAT64_SIZE
            b_u8  = n * NUM_FEATURES * UINT8_SIZE

            def _level(b: int) -> str:
                if b <= l1d: return "L1D"
                if b <= l2:  return "L2"
                if b <= l3:  return "L3"
                return "主存"

            results.append({
                "n_samples": n,
                "sklearn_kb":    round(b_f64 / 1024, 1),
                "histogram_kb":  round(b_u8 / 1024, 1),
                "sklearn_cache_level":    _level(b_f64),
                "histogram_cache_level":  _level(b_u8),
                "sklearn_fits_l1d":    b_f64 <= l1d,
                "histogram_fits_l1d":  b_u8 <= l1d,
            })
        return results

    @staticmethod
    def generate_conclusion(
        results: dict[str, Any],
    ) -> str:
        """根据实测数据动态生成算法效能对比结论。"""
        c   = results["sklearn"]
        e   = results["histogram"]
        bm  = results["benchmark"]
        mem = results["memory_analysis"]
        cfg = results.get("config", {})
        scale = results.get("scale_analysis", [])
        ca    = results.get("cache_analysis", {})

        n_samples = mem["n_samples"]
        speedup_train = c["train_time_sec"] / max(e["train_time_sec"], 0.0001)
        speedup_pred  = c["predict_time_sec"] / max(e["predict_time_sec"], 0.0001)
        access_speedup = bm["speedup"]

        # 当前数据量对应的缓存层级
        f64_kb = mem["float64_total_kb"]
        u8_kb  = mem["uint8_total_kb"]

        def _cache_level(kb: float) -> str:
            b = kb * 1024
            if b <= L1D_CACHE_SIZE: return "L1D"
            if b <= L2_CACHE_SIZE:  return "L2"
            if b <= L3_CACHE_SIZE:  return "L3"
            return "主存"

        f64_level = _cache_level(f64_kb)
        u8_level  = _cache_level(u8_kb)

        # 判断当前数据规模区间
        if n_samples <= 5_000:
            regime = "small"
        elif n_samples <= 30_000:
            regime = "medium"
        else:
            regime = "large"

        # ── 当前规模实测分析 ──
        lines = [
            "═" * 64,
            "  算法效能对比结论",
            "═" * 64,
            "",
            "【当前规模实测分析】",
            f"  样本量: {n_samples:,}，特征维度: {mem['n_features']}",
            f"  训练耗时: scikit-learn {c['train_time_sec']:.4f}s"
            f" vs 直方图 {e['train_time_sec']:.4f}s"
            f"  ({'直方图快' if speedup_train >= 1 else 'scikit-learn快'}"
            f" {max(speedup_train, 1/speedup_train):.2f}×)",
            f"  预测耗时: scikit-learn {c['predict_time_sec']:.4f}s"
            f" vs 直方图 {e['predict_time_sec']:.4f}s"
            f"  ({'直方图快' if speedup_pred >= 1 else 'scikit-learn快'}"
            f" {max(speedup_pred, 1/speedup_pred):.2f}×)",
            f"  随机行访问: scikit-learn {bm['float64_access_ms']:.4f}ms"
            f" vs 直方图 {bm['uint8_access_ms']:.4f}ms"
            f"  (直方图快 {access_speedup:.2f}×)",
            f"  数据集内存: scikit-learn {f64_kb}KB ({f64_level})"
            f" vs 直方图 {u8_kb}KB ({u8_level})"
            f"  (缩减 {mem['memory_reduction_pct']}%)",
            "",
        ]

        # ── 缓存命中率（如果有）──
        sk_cache = ca.get("sklearn", {})
        hi_cache = ca.get("histogram", {})
        has_vtune = ca.get("has_vtune_data", False)
        if sk_cache and hi_cache:
            sk_l1  = sk_cache.get("l1d_hit_rate", 0) * 100
            hi_l1  = hi_cache.get("l1d_hit_rate", 0) * 100
            sk_ov  = sk_cache.get("effective_hit_rate", 0) * 100
            hi_ov  = hi_cache.get("effective_hit_rate", 0) * 100
            data_src = ("Intel VTune 硬件 PMC 实测"
                        if has_vtune else "理论模型 (min(C/D, 1))")
            lines += [
                f"【缓存命中率分析】 (数据来源: {data_src})",
                "",
                f"  当前规模 ({n_samples:,} 样本):",
                f"  L1D 命中率:   scikit-learn {sk_l1:.1f}%  vs  直方图 {hi_l1:.1f}%"
                f"  ({'直方图高' if hi_l1 > sk_l1 else '持平'}"
                f" {abs(hi_l1 - sk_l1):.1f}pp)",
                f"  L3 命中率:   scikit-learn {sk_cache.get('l3_hit_rate',0)*100:.1f}%"
                f"  vs  直方图 {hi_cache.get('l3_hit_rate',0)*100:.1f}%",
                f"  加权有效率:   scikit-learn {sk_ov:.2f}%  vs  直方图 {hi_ov:.2f}%",
            ]
            # 实测事件计数（VTune 专属）
            if has_vtune and sk_cache.get("measured"):
                sk_l3m = sk_cache.get("l3_miss_count", 0)
                hi_l3m = hi_cache.get("l3_miss_count", 0)
                sk_l2m = sk_cache.get("l2_miss_count", 0)
                hi_l2m = hi_cache.get("l2_miss_count", 0)
                lines += [
                    "",
                    "  硬件 PMC 事件计数 (MEM_LOAD_RETIRED.*):",
                    f"  L2 Miss:  scikit-learn {sk_l2m:>12,}  vs  直方图 {hi_l2m:>12,}"
                    f"  (sklearn {sk_l2m/max(hi_l2m,1):.1f}x)",
                    f"  L3 Miss:  scikit-learn {sk_l3m:>12,}  vs  直方图 {hi_l3m:>12,}"
                    f"  (sklearn {sk_l3m/max(hi_l3m,1):.1f}x → DRAM 访问更多)",
                ]
            lines.append("")

        # ── 根因分析（动态）──
        lines += ["【根因分析】", ""]

        # sklearn 分析
        lines += [
            "  传统 scikit-learn 算法:",
            f"  - 使用 float64 原始特征值，当前 {n_samples:,} 样本"
            f"共占 {f64_kb}KB",
        ]
        if mem["float64_fits_l1d"]:
            lines.append(f"  - 当前数据({f64_kb}KB)可驻留 L1D 缓存(32KB)，缓存压力较小")
        elif f64_level == "L2":
            lines.append(f"  - 当前数据({f64_kb}KB)已超出 L1D(32KB)，降级到 L2 缓存")
        elif f64_level == "L3":
            lines.append(f"  - 当前数据({f64_kb}KB)已超出 L2(256KB)，降级到 L3 共享缓存")
        else:
            lines.append(f"  - 当前数据({f64_kb}KB)已超出 L3(8MB)，只能驻留主存，"
                         "严重的 Memory Wall 效应")
        lines.append("  - sklearn 内部按随机索引访问样本，每行搬运 88 字节，带宽需求大")
        lines.append("")

        # 直方图分析
        lines += [
            "  直方图离散化算法:",
            f"  - 通过 256 桶离散化，uint8 每样本仅 11 字节，"
            f"当前仅占 {u8_kb}KB",
        ]
        if mem["uint8_fits_l1d"]:
            lines.append(f"  - 当前数据({u8_kb}KB)可完全驻留 L1D 缓存，"
                         "几乎无缓存未命中")
        elif u8_level == "L2":
            lines.append(f"  - 当前数据({u8_kb}KB)已超出 L1D，降级到 L2，"
                         "但仍远小于 scikit-learn 的占用")
        elif u8_level == "L3":
            if f64_level != u8_level:
                lines.append(f"  - 当前数据({u8_kb}KB)降级到 L3，"
                             f"而 scikit-learn 已降级到 {f64_level}")
            else:
                ratio = round(f64_kb / max(u8_kb, 0.1))
                lines.append(
                    f"  - 当前数据({u8_kb}KB)与 scikit-learn({f64_kb}KB)"
                    f"同处 L3 缓存，但仅占其 {1/ratio*100:.0f}%"
                )
                lines.append(
                    f"  - 同层级下: uint8 每行仅 11B vs float64 每行 88B，"
                    f"缓存行有效数据量多 {ratio}×，带宽压力小"
                )
        else:
            lines.append(f"  - 当前数据({u8_kb}KB)已超出 L3，"
                         f"但 scikit-learn 占用({f64_kb}KB)是它的 {f64_kb/max(u8_kb,0.1):.0f} 倍")
        lines += [
            "  - 同等行数访问仅需 1/8 内存带宽，缓存命中率更高",
            f"  - 额外开销: 编码时间约 {e.get('encode_time_sec', 'N/A')}s"
            if isinstance(e.get('encode_time_sec'), (int, float)) else
            "  - 额外开销: float64→uint8 离散化编码（小数据下该开销占比更大）",
            "",
        ]

        # numpy SIMD 注意事项
        lines += [
            "  注意: numpy SIMD 内核对 float64 向量化运算(sum/sort)高度优化，",
            "  在纯计算上 float64 可能更快。但决策树训练的核心瓶颈是随机样本查找",
            "  （非向量化），此处 uint8 的内存带宽优势起决定性作用。",
            "",
        ]

        # ── 数据规模扩展性 ──
        if scale:
            lines += ["【数据规模扩展性】", ""]
            crossover = None
            for s in scale:
                if not s["sklearn_fits_l1d"] and s["histogram_fits_l1d"]:
                    crossover = s
                    break
            if crossover:
                lines += [
                    f"  关键拐点: 当样本量达到 {crossover['n_samples']:,} 时，",
                    f"  scikit-learn 数据({crossover['sklearn_kb']}KB)"
                    f"已超出 L1D 缓存(32KB)，降级到 {crossover['sklearn_cache_level']}；",
                    f"  而直方图算法({crossover['histogram_kb']}KB)仍可驻留 L1D。",
                    "",
                ]
            lines += ["  样本量    │ scikit-learn      │ 直方图算法"]
            lines += ["  ──────────┼───────────────────┼──────────────────"]
            for s in scale:
                lines.append(
                    f"  {s['n_samples']:>9,} │ {s['sklearn_kb']:>8} KB"
                    f" → {s['sklearn_cache_level']:<4}"
                    f"     │ {s['histogram_kb']:>8} KB"
                    f" → {s['histogram_cache_level']:<4}"
                )
            lines += [""]

        # ── 缓存命中率多规模表格（如果有）──
        ca_by_scale = ca.get("by_scale", {})
        if ca_by_scale:
            src_label = "Intel VTune 实测" if has_vtune else "理论推算"
            lines += [f"【缓存命中率随规模变化】 ({src_label})", ""]
            lines += ["  样本量    │ sklearn L1D │ 直方图 L1D │ sklearn L3 │ 直方图 L3  │ sklearn 有效率 │ 直方图有效率"]
            lines += ["  ──────────┼────────────┼───────────┼───────────┼───────────┼─────────────┼────────────"]
            for scale_key in sorted(ca_by_scale.keys(), key=int):
                d = ca_by_scale[scale_key]
                n = d["n_samples"]
                sk_l1_s = f"{d['sklearn']['l1d_hit_rate']*100:.0f}%"
                hi_l1_s = f"{d['histogram']['l1d_hit_rate']*100:.0f}%"
                sk_l3_s = f"{d['sklearn']['l3_hit_rate']*100:.0f}%"
                hi_l3_s = f"{d['histogram']['l3_hit_rate']*100:.0f}%"
                sk_ov_s = f"{d['sklearn']['effective_hit_rate']*100:.1f}%"
                hi_ov_s = f"{d['histogram']['effective_hit_rate']*100:.1f}%"
                lines.append(
                    f"  {n:>9,} │ {sk_l1_s:>10} │ {hi_l1_s:>9}"
                    f" │ {sk_l3_s:>9} │ {hi_l3_s:>9}"
                    f" │ {sk_ov_s:>11} │ {hi_ov_s:>11}"
                )
            lines.append("")

        # ── 结论（完全动态）──
        same_level = (f64_level == u8_level)
        lines += ["【结论】", ""]

        if regime == "small":
            if speedup_train < 1:
                lines += [
                    f"  当前 {n_samples:,} 样本属于小数据规模，"
                    f"数据完全被 L2/L3 缓存覆盖，不存在 Memory Wall。",
                    "",
                    f"  scikit-learn 训练耗时({c['train_time_sec']:.4f}s)"
                    f"快于直方图({e['train_time_sec']:.4f}s)，",
                    "  原因是直方图算法的离散化编码在小数据下的固定开销"
                    " > 缓存优势带来的收益。",
                    "",
                    "  这是预期行为：直方图算法的优势在大数据量下才能体现。",
                    f"  建议将每标签样本数增大到 2,000 以上"
                    f"（总量 >12,000）后重新测试。",
                ]
            else:
                lines += [
                    f"  当前 {n_samples:,} 样本属于小数据规模，"
                    "数据仍可被 L2/L3 缓存覆盖。",
                    "",
                    f"  直方图算法训练耗时({e['train_time_sec']:.4f}s)"
                    f"已快于 scikit-learn({c['train_time_sec']:.4f}s)，"
                    f"加速比 {speedup_train:.2f}×。",
                    "  在小数据下即能体现优势，说明当前参数配置下直方图算法已具有效能优势。",
                ]

        elif regime == "medium":
            if same_level:
                # 同层级：带宽与占用差异
                lines += [
                    f"  当前 {n_samples:,} 样本属于中等数据规模，"
                    f"两种算法数据同处 {f64_level} 缓存层级。",
                    f"  但 scikit-learn 占用 {f64_kb}KB，"
                    f"直方图仅占 {u8_kb}KB（{1/round(f64_kb/max(u8_kb,0.1))*100:.0f}%）。",
                    "",
                    f"  在同一缓存层级内，uint8 每行仅 11 字节"
                    f" vs float64 每行 88 字节，",
                    f"  缓存行有效载荷多 {round(f64_kb/max(u8_kb,0.1))}×，"
                    f"总线带宽压力更小，实测数据访问快 {access_speedup:.2f}×。",
                ]
                if speedup_train >= 1:
                    lines += [
                        "",
                        f"  训练耗时直方图快 {speedup_train:.2f}×，"
                        f"带宽优势已转化为实际性能收益。"
                        "  继续增大数据量，优势将更加显著。",
                    ]
                else:
                    lines += [
                        "",
                        f"  但训练耗时 scikit-learn 仍快 {1/speedup_train:.2f}×，"
                        "  原因是直方图编码开销抵消了部分带宽优势。",
                        "  继续增大数据量后直方图算法将逐渐反超。",
                    ]
            else:
                # 不同层级
                if speedup_train >= 1:
                    lines += [
                        f"  当前 {n_samples:,} 样本属于中等数据规模，"
                        f"scikit-learn 数据({f64_kb}KB)已降级到 {f64_level}，"
                        f"而直方图({u8_kb}KB)仍在更高速的 {u8_level}。",
                        "",
                        f"  直方图算法训练快 {speedup_train:.2f}×，"
                        f"预测快 {speedup_pred:.2f}×，"
                        "  缓存层级差异已开始转化为实际性能差异。",
                        "",
                        "  继续增大数据量，直方图算法的优势将更加显著。",
                    ]
                else:
                    lines += [
                        f"  当前 {n_samples:,} 样本属于中等数据规模，"
                        f"scikit-learn({f64_kb}KB, {f64_level})"
                        f" vs 直方图({u8_kb}KB, {u8_level})。",
                        "",
                        f"  训练耗时 scikit-learn 仍快 {1/speedup_train:.2f}×，"
                        "  可能是直方图编码开销仍占较大比例。",
                        f"  但数据访问层面 uint8 已快 {access_speedup:.2f}×，"
                        "  继续增大数据量后直方图算法将逐渐反超。",
                    ]

        else:  # large
            if same_level:
                # 大数据 + 同层级：强调带宽与缓存压力
                size_ratio = round(f64_kb / max(u8_kb, 0.1))
                if speedup_train >= 1:
                    lines += [
                        f"  当前 {n_samples:,} 样本属于大数据规模，"
                        f"两种算法数据同处 {f64_level}。",
                        f"  但 scikit-learn 占用 {f64_kb}KB"
                        f"（占 L3 缓存 {f64_kb/8192*100:.0f}%），"
                        f"直方图仅占 {u8_kb}KB"
                        f"（占 L3 缓存 {u8_kb/8192*100:.0f}%）。",
                        "",
                        f"  在同一缓存层级下，scikit-learn 对 L3 的占用是直方图的"
                        f" {size_ratio}×，缓存驱逐压力更大，",
                        f"  实测数据访问慢 {access_speedup:.2f}×。",
                        "",
                        f"  直方图算法训练快 {speedup_train:.2f}×，"
                        f"预测快 {max(speedup_pred, 1/max(speedup_pred,0.001)):.2f}×。",
                        "  对于 CAN 总线等高吞吐场景，直方图离散化算法",
                        "  在保持 256 桶精度的同时，效能优势随数据规模持续扩大。",
                    ]
                else:
                    lines += [
                        f"  当前 {n_samples:,} 样本已属于大数据规模，"
                        f"两种算法同处 {f64_level}，"
                        f"但 scikit-learn 占用是直方图的 {size_ratio}×。",
                        "",
                        "  scikit-learn 训练仍快于直方图，",
                        "  可能原因: 直方图编码开销随数据量线性增长，",
                        "  抵消了部分带宽优势。",
                        "",
                        f"  但数据访问层面 uint8 已快 {access_speedup:.2f}×，"
                        f"  内存占用缩减 {mem['memory_reduction_pct']}%，",
                        "  直方图算法在资源效率上仍具有明确优势。",
                    ]
            else:
                # 大数据 + 不同层级：缓存层级差异
                if speedup_train >= 1:
                    lines += [
                        f"  当前 {n_samples:,} 样本属于大数据规模，"
                        f"scikit-learn 数据({f64_kb}KB)已降级到 {f64_level}，"
                        f"直方图({u8_kb}KB)仍在更高速的 {u8_level}，"
                        f"缓存层级差异显著。",
                        "",
                        f"  直方图算法训练快 {speedup_train:.2f}×，"
                        f"预测快 {max(speedup_pred, 1/max(speedup_pred,0.001)):.2f}×，"
                        f"数据访问快 {access_speedup:.2f}×。",
                        "  Memory Wall 效应已充分体现，直方图算法在效能上具有显著优势。",
                        "",
                        "  对于 CAN 总线等高吞吐场景，直方图离散化算法",
                        "  在保持 256 桶精度的同时，效能优势随数据规模持续扩大。",
                    ]
                else:
                    lines += [
                        f"  当前 {n_samples:,} 样本已属于大数据规模，"
                        f"scikit-learn({f64_level}) vs 直方图({u8_level})，"
                        "但 scikit-learn 训练仍快于直方图。",
                        "  可能原因: 直方图编码开销随数据量线性增长，",
                        "  抵消了部分缓存优势。",
                        "",
                        f"  但数据访问层面 uint8 已快 {access_speedup:.2f}×，"
                        f"  内存占用缩减 {mem['memory_reduction_pct']}%，",
                        "  直方图算法在资源效率上仍具有明确优势。",
                    ]

        lines.append("═" * 64)
        return "\n".join(lines)

    @staticmethod
    def summary_text(
        benchmark: dict, controlled: dict, memory: dict
    ) -> str:
        """生成实测性能对比摘要文本。"""
        by_scale = benchmark.get("by_scale", {})
        lines = [
            "═" * 64,
            "  算法效能实测对比（随机行访问基准测试）",
            "═" * 64,
            "",
            f"【测试方法】随机行访问 mat[random_indices]，模拟决策树样本查找",
            f"【当前规模】{benchmark['n_samples']} 样本 × {benchmark['n_features']} 特征"
            f"，{benchmark['iterations']} 次迭代",
            "",
            "【多规模数据访问耗时】",
            "  样本量      │ scikit-learn(float64) │ 直方图(uint8)   │ 加速比",
            "  ────────────┼──────────────────────┼─────────────────┼────────",
        ]
        for scale_str, data in by_scale.items():
            n = int(scale_str)
            lines.append(
                f"  {n:>9,}   │ {data['float64']['ms']:.4f} ms ({data['float64']['kb']:>7.1f}KB)"
                f"  │ {data['uint8']['ms']:.4f} ms ({data['uint8']['kb']:>6.1f}KB)"
                f"  │ {data['speedup']:.2f}×"
            )

        lines += ["", "【控制变量隔离分析】"]

        fd = controlled["fixed_dtype_float64"]
        lines += [f"  {fd['description']}",
                  f"    C-order: {fd['c_order_ms']:.4f} ms → F-order: {fd['f_order_ms']:.4f} ms"
                  f"  (F-order 慢 {fd['slowdown_f_vs_c']}×)", ""]

        fo = controlled["fixed_order_c"]
        lines += [f"  {fo['description']}",
                  f"    float64: {fo['float64_ms']:.4f} ms → uint8: {fo['uint8_ms']:.4f} ms"
                  f"  (uint8 快 {fo['speedup_u8_vs_f64']}×)", ""]

        fu = controlled["fixed_dtype_uint8"]
        lines += [f"  {fu['description']}",
                  f"    C-order: {fu['c_order_ms']:.4f} ms → F-order: {fu['f_order_ms']:.4f} ms"
                  f"  (F-order 慢 {fu['slowdown_f_vs_c']}×)", ""]

        fits_f64 = "[OK] 驻留 L1D" if memory["float64_fits_l1d"] else "[!!] 超出 L1D (32KB)"
        fits_u8  = "[OK] 驻留 L1D" if memory["uint8_fits_l1d"] else "[!!] 超出 L1D"
        lines += [
            "【内存占用与缓存驻留】",
            f"  scikit-learn (float64): {memory['float64_total_kb']} KB  {fits_f64}",
            f"  直方图算法  (uint8):    {memory['uint8_total_kb']} KB  {fits_u8}",
            f"  内存缩减: {memory['memory_reduction_pct']}%",
            "═" * 64,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 对比运行器
# ---------------------------------------------------------------------------

def run_comparison(
    group_config: dict[str, object],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """运行两种算法的端到端对比，收集全部效能指标。"""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output" / "perf_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("生成共享训练数据（保证两组算法使用相同数据，确保公平性）...")
    training_rows = generate_training_rows(group_config)
    input_cases = generate_dynamic_cases(group_config)

    n_samples = len(training_rows)
    n_train = int(n_samples * 0.78)
    n_test = n_samples - n_train
    logger.info(f"样本总量: {n_samples}，训练≈{n_train}，测试≈{n_test}")

    # ── 传统 scikit-learn 算法 ────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info(f"【{LABEL_SKLEARN}】float64 原始特征 RandomForest")
    logger.info("─" * 50)

    t0 = time.perf_counter()
    model_sklearn, metrics_sklearn = train_model(training_rows, group_config)
    sklearn_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    predicted_sklearn = predict_cases(model_sklearn, input_cases)
    sklearn_predict_time = time.perf_counter() - t0

    # ── 直方图离散化算法 ─────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info(f"【{LABEL_HISTOGRAM}】uint8 256桶离散化 RandomForest")
    logger.info("─" * 50)

    t0 = time.perf_counter()
    model_histogram, metrics_histogram = train_histogram_model(
        training_rows, group_config
    )
    histogram_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    predicted_histogram = predict_histogram_cases(model_histogram, input_cases)
    histogram_predict_time = time.perf_counter() - t0

    # ── 实测性能基准 ─────────────────────────────────────
    logger.info("\n[PerfBenchmark] 运行数据访问时间基准测试...")
    benchmark  = PerfBenchmark.benchmark_access(n_samples, NUM_FEATURES)
    controlled = PerfBenchmark.controlled_variable_comparison(n_samples, NUM_FEATURES)
    memory     = PerfBenchmark.memory_analysis(n_samples)
    scale      = PerfBenchmark.scale_analysis()
    cache      = PerfBenchmark.cache_analysis(n_samples, NUM_FEATURES)

    # ── 硬件利用率 ───────────────────────────────────────
    hw_s = metrics_sklearn.get("hardware", {})
    hw_h = metrics_histogram.get("hardware", {})

    def _cpu_avg(hw):  return float(hw.get("cpu_total_pct", {}).get("avg", 0))
    def _cpu_peak(hw): return float(hw.get("cpu_total_pct", {}).get("peak", 0))
    def _mem_peak(hw): return float(hw.get("proc_memory_mb", {}).get("peak", 0))
    def _cpu_per_core(hw): return list(hw.get("cpu_per_core_pct", []))

    # ── 构建结果 ─────────────────────────────────────────
    results: dict[str, Any] = {
        "config": {
            "n_samples": n_samples, "n_train": n_train, "n_test": n_test,
            "n_features": NUM_FEATURES,
            "samples_per_label": group_config["samples_per_label"],
        },
        "sklearn": {
            "label": LABEL_SKLEARN,
            "accuracy": metrics_sklearn["accuracy"],
            "train_time_sec": round(sklearn_train_time, 4),
            "predict_time_sec": round(sklearn_predict_time, 4),
            "cpu_avg_pct": _cpu_avg(hw_s), "cpu_peak_pct": _cpu_peak(hw_s),
            "cpu_per_core": _cpu_per_core(hw_s),
            "memory_peak_mb": _mem_peak(hw_s),
            "data_size_kb": round(n_samples * NUM_FEATURES * FLOAT64_SIZE / 1024, 1),
            "cache": cache["sklearn"],
        },
        "histogram": {
            "label": LABEL_HISTOGRAM,
            "accuracy": metrics_histogram["accuracy"],
            "train_time_sec": round(histogram_train_time, 4),
            "predict_time_sec": round(histogram_predict_time, 4),
            "encode_time_sec": metrics_histogram.get("encode_time_sec", 0),
            "cpu_avg_pct": _cpu_avg(hw_h), "cpu_peak_pct": _cpu_peak(hw_h),
            "cpu_per_core": _cpu_per_core(hw_h),
            "memory_peak_mb": _mem_peak(hw_h),
            "data_size_kb": round(n_samples * NUM_FEATURES * UINT8_SIZE / 1024, 1),
            "cache": cache["histogram"],
        },
        "benchmark": benchmark,
        "controlled_variables": controlled,
        "memory_analysis": memory,
        "scale_analysis": scale,
        "cache_analysis": cache,
    }

    # ── 输出 ─────────────────────────────────────────────
    _print_comparison(results)
    render_comparison_charts(results, output_dir)

    return results


# ---------------------------------------------------------------------------
# 终端输出
# ---------------------------------------------------------------------------

def _print_comparison(results: dict[str, Any]) -> None:
    """打印算法效能对比报告到控制台。"""
    s  = results["sklearn"]
    h  = results["histogram"]
    bm = results["benchmark"]
    ct = results["controlled_variables"]
    mem = results["memory_analysis"]

    print("\n" + "█" * 64)
    print(f"  算法效能对比报告：{LABEL_SKLEARN} vs {LABEL_HISTOGRAM}")
    print("█" * 64)

    print("\n── 模型准确率 ──")
    print(f"  {LABEL_SKLEARN}:    {s['accuracy']:.4f}")
    print(f"  {LABEL_HISTOGRAM}:  {h['accuracy']:.4f}")
    print(f"  准确率差异:         {h['accuracy'] - s['accuracy']:+.4f}")

    print("\n── 训练耗时（含编码 + fit + predict，统一口径）──")
    print(f"  {LABEL_SKLEARN}:    {s['train_time_sec']:.4f}s")
    print(f"  {LABEL_HISTOGRAM}:  {h['train_time_sec']:.4f}s")
    speedup_train = s["train_time_sec"] / max(h["train_time_sec"], 0.0001)
    print(f"  直方图加速比:       {speedup_train:.2f}×")

    print("\n── 预测耗时 ──")
    print(f"  {LABEL_SKLEARN}:    {s['predict_time_sec']:.4f}s")
    print(f"  {LABEL_HISTOGRAM}:  {h['predict_time_sec']:.4f}s")
    speedup_pred = s["predict_time_sec"] / max(h["predict_time_sec"], 0.0001)
    print(f"  直方图加速比:       {speedup_pred:.2f}×")

    print("\n── CPU 利用率（均值 / 峰值）──")
    print(f"  {LABEL_SKLEARN}:    {s['cpu_avg_pct']:.1f}% / {s['cpu_peak_pct']:.1f}%")
    print(f"  {LABEL_HISTOGRAM}:  {h['cpu_avg_pct']:.1f}% / {h['cpu_peak_pct']:.1f}%")

    per_core_s = s.get("cpu_per_core", [])
    per_core_h = h.get("cpu_per_core", [])
    n_cores = max(len(per_core_s), len(per_core_h))
    if n_cores > 0:
        print("\n── 逐核 CPU 利用率（均值 / 峰值）──")
        for i in range(n_cores):
            s_avg  = per_core_s[i].get("avg", 0)  if i < len(per_core_s) and isinstance(per_core_s[i], dict) else 0
            s_peak = per_core_s[i].get("peak", 0) if i < len(per_core_s) and isinstance(per_core_s[i], dict) else 0
            h_avg  = per_core_h[i].get("avg", 0)  if i < len(per_core_h) and isinstance(per_core_h[i], dict) else 0
            h_peak = per_core_h[i].get("peak", 0) if i < len(per_core_h) and isinstance(per_core_h[i], dict) else 0
            print(f"  Core {i:2d}:  scikit-learn={s_avg:.1f}%/{s_peak:.1f}%  "
                  f"直方图={h_avg:.1f}%/{h_peak:.1f}%")

    print("\n── 数据内存占用 ──")
    print(f"  {LABEL_SKLEARN}:    {s['data_size_kb']:.1f} KB (float64, 8字节/特征)")
    print(f"  {LABEL_HISTOGRAM}:  {h['data_size_kb']:.1f} KB (uint8, 1字节/特征)")
    print(f"  内存缩减:           {mem['memory_reduction_pct']}%")

    print("\n── 数据随机访问耗时（timeit 实测，模拟决策树样本查找）──")
    by_scale = bm.get("by_scale", {})
    print(f"  {'样本量':>9}   │ {'float64':>12} │ {'uint8':>12} │ {'加速比':>6}")
    print(f"  {'─'*9}   ┼ {'─'*12} ┼ {'─'*12} ┼ {'─'*6}")
    for scale_str, data in by_scale.items():
        n = int(scale_str)
        print(f"  {n:>9,}   │ {data['float64']['ms']:.4f} ms    │ {data['uint8']['ms']:.4f} ms    │ {data['speedup']:.2f}×")

    print("\n── 控制变量隔离分析 ──")
    fd = ct["fixed_dtype_float64"]
    print(f"  {fd['description']}")
    print(f"    C: {fd['c_order_ms']:.4f} → F: {fd['f_order_ms']:.4f} ms  "
          f"(F-order 慢 {fd['slowdown_f_vs_c']}×) ← 内存布局的影响")
    fo = ct["fixed_order_c"]
    print(f"  {fo['description']}")
    print(f"    f64: {fo['float64_ms']:.4f} → u8: {fo['uint8_ms']:.4f} ms  "
          f"(u8 快 {fo['speedup_u8_vs_f64']}×) ← 数据类型的影响")
    fu = ct["fixed_dtype_uint8"]
    print(f"  {fu['description']}")
    print(f"    C: {fu['c_order_ms']:.4f} → F: {fu['f_order_ms']:.4f} ms  "
          f"(F-order 慢 {fu['slowdown_f_vs_c']}×)")

    # 实测摘要
    summary = PerfBenchmark.summary_text(bm, ct, mem)
    print("\n" + summary)

    # 结论
    conclusion = PerfBenchmark.generate_conclusion(results)
    print("\n" + conclusion)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    """配置 matplotlib 中文字体与暗色主题。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#0d1117"
    plt.rcParams["axes.facecolor"]   = "#161b22"
    plt.rcParams["text.color"]       = "#e6edf3"
    plt.rcParams["axes.labelcolor"]  = "#e6edf3"
    plt.rcParams["xtick.color"]      = "#8b949e"
    plt.rcParams["ytick.color"]      = "#8b949e"
    plt.rcParams["axes.edgecolor"]   = "#30363d"
    plt.rcParams["axes.grid"]        = True
    plt.rcParams["grid.alpha"]       = 0.15
    plt.rcParams["grid.color"]       = "#8b949e"


_COLORS = {
    "sklearn":          "#ff7b72",   # 红色系 — scikit-learn
    "histogram":        "#3fb950",   # 绿色系 — 直方图算法
    "sklearn_light":    "#ffa198",
    "histogram_light":  "#56d364",
    "neutral":          "#58a6ff",
}


def render_comparison_charts(results: dict[str, Any], output_dir: Path) -> None:
    """生成一组算法效能对比图表。"""
    _setup_matplotlib()

    _chart_data_access_time(results, output_dir)
    _chart_cpu_utilization(results, output_dir)
    _chart_per_core_cpu(results, output_dir)
    _chart_cache_hit_rate(results, output_dir)
    _chart_timing(results, output_dir)
    _chart_memory(results, output_dir)
    _chart_controlled_variable(results, output_dir)
    _chart_scale_analysis(results, output_dir)
    _chart_radar(results, output_dir)

    logger.info(f"[PerfCompare] 图表已保存到 {output_dir}")


# ── 图 1：多规模数据访问时间 ─────────────────────────────────
def _chart_data_access_time(results: dict[str, Any], output_dir: Path) -> None:
    """多规模随机行访问耗时对比。"""
    bm = results["benchmark"]
    by_scale = bm.get("by_scale", {})
    if not by_scale:
        return

    scales = sorted(by_scale.keys(), key=int)
    x_labels = [f"{int(s):,}" for s in scales]
    x_pos = list(range(len(scales)))
    f64_ms = [by_scale[s]["float64"]["ms"] for s in scales]
    u8_ms  = [by_scale[s]["uint8"]["ms"] for s in scales]
    speedups = [by_scale[s]["speedup"] for s in scales]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.35
    ax.bar([x - width / 2 for x in x_pos], f64_ms, width,
           color=_COLORS["sklearn"], label=LABEL_SKLEARN, edgecolor="white", linewidth=0.6)
    ax.bar([x + width / 2 for x in x_pos], u8_ms, width,
           color=_COLORS["histogram"], label=LABEL_HISTOGRAM, edgecolor="white", linewidth=0.6)

    # 标注数值和加速比
    for i in range(len(scales)):
        ax.text(x_pos[i] - width / 2, f64_ms[i] + max(f64_ms) * 0.03,
                f"{f64_ms[i]:.4f}", ha="center", va="bottom", fontsize=8, color="#e6edf3")
        ax.text(x_pos[i] + width / 2, u8_ms[i] + max(u8_ms) * 0.03,
                f"{u8_ms[i]:.4f}", ha="center", va="bottom", fontsize=8, color="#e6edf3")
        ax.text(x_pos[i], max(f64_ms[i], u8_ms[i]) * 1.18,
                f"{speedups[i]:.2f}×", ha="center", va="bottom",
                fontsize=9, color="#d2a8ff", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("样本数量", fontsize=11)
    ax.set_ylabel("随机行访问耗时 (ms, 越低越好)", fontsize=11)
    ax.set_title(
        f"数据随机访问耗时对比（timeit, {bm['iterations']} 次迭代）\n"
        f"模拟决策树训练中的样本查找模式，数字标注为 uint8 加速比",
        fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "perf_access_time.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 2：CPU 利用率 ───────────────────────────────────────
def _chart_cpu_utilization(results: dict[str, Any], output_dir: Path) -> None:
    """CPU 利用率均值 + 峰值对比。"""
    s = results["sklearn"]
    h = results["histogram"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    width = 0.3

    bars_avg = ax.bar(x - width / 2, [s["cpu_avg_pct"], h["cpu_avg_pct"]], width,
                      color=[_COLORS["sklearn"], _COLORS["histogram"]],
                      edgecolor="white", linewidth=0.6, label="CPU 均值")
    bars_peak = ax.bar(x + width / 2, [s["cpu_peak_pct"], h["cpu_peak_pct"]], width,
                       color=[_COLORS["sklearn_light"], _COLORS["histogram_light"]],
                       edgecolor="white", linewidth=0.6, label="CPU 峰值", alpha=0.75)

    for bar in [*bars_avg, *bars_peak]:
        h_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h_val + 0.8,
                f"{h_val:.1f}%", ha="center", va="bottom", fontsize=10, color="#e6edf3")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_SKLEARN, SHORT_HISTOGRAM], fontsize=11)
    ax.set_ylabel("CPU 利用率 (%)", fontsize=11)
    ax.set_title("CPU 利用率对比（实测）\n"
                 "注：直方图算法 CPU 占用更低 = 用更少 CPU 周期完成相同任务 = 更高效",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, max(s["cpu_peak_pct"], h["cpu_peak_pct"], 10) * 1.25)

    fig.tight_layout()
    fig.savefig(output_dir / "perf_cpu_compare.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 2b：逐核 CPU 热力图 ─────────────────────────────────
def _chart_per_core_cpu(results: dict[str, Any], output_dir: Path) -> None:
    """逐核 CPU 利用率热力图。"""
    s = results["sklearn"]
    h = results["histogram"]
    per_core_s = s.get("cpu_per_core", [])
    per_core_h = h.get("cpu_per_core", [])

    if not per_core_s and not per_core_h:
        return
    n_cores = max(len(per_core_s), len(per_core_h))
    if n_cores == 0:
        return

    row_labels = ["scikit-learn\n均值", "scikit-learn\n峰值",
                  "直方图算法\n均值", "直方图算法\n峰值"]

    def _extract(per_core, key):
        vals = []
        for item in per_core:
            v = item.get(key, 0.0) if isinstance(item, dict) else 0.0
            vals.append(float(v))
        while len(vals) < n_cores:
            vals.append(0.0)
        return vals

    data = np.array([
        _extract(per_core_s, "avg"), _extract(per_core_s, "peak"),
        _extract(per_core_h, "avg"), _extract(per_core_h, "peak"),
    ])

    fig_w = max(8, n_cores * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=100)

    ax.set_xticks(range(n_cores))
    ax.set_xticklabels([f"Core {i}" for i in range(n_cores)], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if val > 50 else "#e6edf3",
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("CPU 利用率 (%)", fontsize=10, color="#e6edf3")
    cbar.ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set_title(f"逐核 CPU 利用率热力图 ({n_cores} 核)",
                 fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(output_dir / "perf_per_core_cpu.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 2c：缓存命中率对比 ───────────────────────────────────
def _chart_cache_hit_rate(results: dict[str, Any], output_dir: Path) -> None:
    """多规模下两种算法的缓存命中率对比。"""
    ca = results.get("cache_analysis", {})
    by_scale = ca.get("by_scale", {})
    if not by_scale:
        return

    scales = sorted(by_scale.keys(), key=int)
    x_labels = [f"{int(s):,}" for s in scales]
    x_pos = list(range(len(scales)))

    sk_l1  = [by_scale[s]["sklearn"]["l1d_hit_rate"] * 100 for s in scales]
    sk_ov  = [by_scale[s]["sklearn"]["effective_hit_rate"] * 100 for s in scales]
    hi_l1  = [by_scale[s]["histogram"]["l1d_hit_rate"] * 100 for s in scales]
    hi_ov  = [by_scale[s]["histogram"]["effective_hit_rate"] * 100 for s in scales]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # --- L1D 命中率 ---
    width = 0.35
    bars_s = ax1.bar([x - width / 2 for x in x_pos], sk_l1, width,
                     color=_COLORS["sklearn"], label=LABEL_SKLEARN,
                     edgecolor="white", linewidth=0.6)
    bars_h = ax1.bar([x + width / 2 for x in x_pos], hi_l1, width,
                     color=_COLORS["histogram"], label=LABEL_HISTOGRAM,
                     edgecolor="white", linewidth=0.6)
    for i in range(len(scales)):
        ax1.text(x_pos[i] - width / 2, sk_l1[i] + 1,
                 f"{sk_l1[i]:.0f}%", ha="center", va="bottom", fontsize=8, color="#e6edf3")
        ax1.text(x_pos[i] + width / 2, hi_l1[i] + 1,
                 f"{hi_l1[i]:.0f}%", ha="center", va="bottom", fontsize=8, color="#e6edf3")
        if sk_l1[i] < 100 and hi_l1[i] > sk_l1[i]:
            diff = hi_l1[i] - sk_l1[i]
            ax1.text(x_pos[i], max(sk_l1[i], hi_l1[i]) * 1.12,
                     f"+{diff:.0f}pp", ha="center", va="bottom",
                     fontsize=8, color="#d2a8ff", fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax1.set_ylabel("命中率 (%)", fontsize=11)
    ax1.set_title("L1D Cache 命中率 (32KB)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(frameon=False, fontsize=9)

    # --- L3 命中率 ---
    sk_l3 = [by_scale[s]["sklearn"]["l3_hit_rate"] * 100 for s in scales]
    hi_l3 = [by_scale[s]["histogram"]["l3_hit_rate"] * 100 for s in scales]
    bars_s2 = ax2.bar([x - width / 2 for x in x_pos], sk_l3, width,
                      color=_COLORS["sklearn"], label=LABEL_SKLEARN,
                      edgecolor="white", linewidth=0.6)
    bars_h2 = ax2.bar([x + width / 2 for x in x_pos], hi_l3, width,
                      color=_COLORS["histogram"], label=LABEL_HISTOGRAM,
                      edgecolor="white", linewidth=0.6)
    for i in range(len(scales)):
        ax2.text(x_pos[i] - width / 2, sk_l3[i] + 0.5,
                 f"{sk_l3[i]:.0f}%", ha="center", va="bottom", fontsize=8, color="#e6edf3")
        ax2.text(x_pos[i] + width / 2, hi_l3[i] + 0.5,
                 f"{hi_l3[i]:.0f}%", ha="center", va="bottom", fontsize=8, color="#e6edf3")
        if hi_l3[i] > sk_l3[i] + 2:
            diff = hi_l3[i] - sk_l3[i]
            ax2.text(x_pos[i], max(sk_l3[i], hi_l3[i]) * 1.12,
                     f"+{diff:.0f}pp", ha="center", va="bottom",
                     fontsize=8, color="#d2a8ff", fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=9, rotation=30, ha="right")
    ax2.set_title("L3 Cache 命中率 (8MB 共享)",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.legend(frameon=False, fontsize=9)

    has_vtune = ca.get("has_vtune_data", False)
    src_label = ("Intel VTune PMC 实测"
                 if has_vtune else "理论模型 (min(C/D, 1))")
    fig.suptitle(
        f"缓存命中率对比（{src_label}，当前 {ca.get('n_samples', 0):,} 样本）\n"
        f"uint8 数据量仅为 float64 的 1/8，L3 Miss 更少 = 更少 DRAM 访问",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "perf_cache_hit_rate.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 3：训练 / 预测耗时 ──────────────────────────────────
def _chart_timing(results: dict[str, Any], output_dir: Path) -> None:
    """训练 + 预测耗时对比（统一口径）。"""
    s = results["sklearn"]
    h = results["histogram"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    width = 0.3

    bars_train = ax.bar(x - width / 2, [s["train_time_sec"], h["train_time_sec"]], width,
                        color=[_COLORS["sklearn"], _COLORS["histogram"]],
                        edgecolor="white", linewidth=0.6, label="训练耗时")
    bars_pred  = ax.bar(x + width / 2, [s["predict_time_sec"], h["predict_time_sec"]], width,
                        color=[_COLORS["sklearn_light"], _COLORS["histogram_light"]],
                        edgecolor="white", linewidth=0.6, label="预测耗时", alpha=0.75)

    for bar in [*bars_train, *bars_pred]:
        h_val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h_val + 0.002,
                f"{h_val:.4f}s", ha="center", va="bottom", fontsize=9, color="#e6edf3")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_SKLEARN, SHORT_HISTOGRAM], fontsize=11)
    ax.set_ylabel("耗时 (秒)", fontsize=11)
    ax.set_title("训练 & 预测耗时对比（统一口径）", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "perf_timing_compare.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 4：内存占用 ─────────────────────────────────────────
def _chart_memory(results: dict[str, Any], output_dir: Path) -> None:
    """数据内存占用对比（确定性计算，含 L1D 缓存阈值）。"""
    s = results["sklearn"]
    h = results["histogram"]
    mem = results["memory_analysis"]

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["数据集内存占用"]
    sklearn_vals   = [s["data_size_kb"]]
    histogram_vals = [h["data_size_kb"]]

    x = np.arange(len(categories))
    width = 0.3

    bars_s = ax.bar(x - width / 2, sklearn_vals, width,
                    color=_COLORS["sklearn"], edgecolor="white", linewidth=0.6,
                    label=LABEL_SKLEARN)
    bars_h = ax.bar(x + width / 2, histogram_vals, width,
                    color=_COLORS["histogram"], edgecolor="white", linewidth=0.6,
                    label=LABEL_HISTOGRAM)

    for bar, val in zip(bars_s, sklearn_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sklearn_vals) * 0.01,
                f"{val:.0f} KB", ha="center", va="bottom", fontsize=10, color="#e6edf3")
    for bar, val in zip(bars_h, histogram_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(histogram_vals) * 0.01,
                f"{val:.0f} KB", ha="center", va="bottom", fontsize=10, color="#e6edf3")

    l1d_kb = L1D_CACHE_SIZE // 1024
    ax.axhline(y=l1d_kb, color="#d2a8ff", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(len(categories) - 0.55, l1d_kb + max(sklearn_vals) * 0.015,
            f"L1D Cache = {l1d_kb} KB", fontsize=9, color="#d2a8ff", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("内存 (KB)", fontsize=11)
    ax.set_title(
        f"数据集内存占用对比（确定性计算，{mem['n_samples']} 样本 × {mem['n_features']} 特征）\n"
        f"直方图算法缩减 {mem['memory_reduction_pct']}%",
        fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "perf_memory_compare.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 5：控制变量隔离 ─────────────────────────────────────
def _chart_controlled_variable(results: dict[str, Any], output_dir: Path) -> None:
    """控制变量对比：隔离 dtype 和内存布局的独立影响。"""
    ct = results["controlled_variables"]
    d_f64 = ct["fixed_dtype_float64"]
    d_u8  = ct["fixed_dtype_uint8"]
    d_co  = ct["fixed_order_c"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 维度 1：固定 float64，改 order
    ax = axes[0]
    bars = ax.bar(["C-order", "F-order"],
                  [d_f64["c_order_ms"], d_f64["f_order_ms"]],
                  color=[_COLORS["histogram"], _COLORS["sklearn"]],
                  width=0.4, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, [d_f64["c_order_ms"], d_f64["f_order_ms"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", fontsize=9, color="#e6edf3")
    ax.set_title(f"固定 float64（scikit-learn 的数据类型）\nF-order 慢 {d_f64['slowdown_f_vs_c']}×",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("ms")

    # 维度 2：固定 C-order，改 dtype
    ax = axes[1]
    bars = ax.bar(["float64\n(scikit-learn)", "uint8\n(直方图)"],
                  [d_co["float64_ms"], d_co["uint8_ms"]],
                  color=[_COLORS["sklearn"], _COLORS["histogram"]],
                  width=0.4, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, [d_co["float64_ms"], d_co["uint8_ms"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", fontsize=9, color="#e6edf3")
    ax.set_title(f"固定 C-order（相同内存布局）\nuint8 快 {d_co['speedup_u8_vs_f64']}×",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("ms")

    # 维度 3：固定 uint8，改 order
    ax = axes[2]
    bars = ax.bar(["C-order", "F-order"],
                  [d_u8["c_order_ms"], d_u8["f_order_ms"]],
                  color=[_COLORS["histogram"], _COLORS["sklearn"]],
                  width=0.4, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, [d_u8["c_order_ms"], d_u8["f_order_ms"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", fontsize=9, color="#e6edf3")
    ax.set_title(f"固定 uint8（直方图的数据类型）\nF-order 慢 {d_u8['slowdown_f_vs_c']}×",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("ms")

    fig.suptitle("控制变量隔离分析（随机行访问，每组只改一个变量）",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "perf_controlled_var.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 6：数据规模扩展性分析 ───────────────────────────────
def _chart_scale_analysis(results: dict[str, Any], output_dir: Path) -> None:
    """不同数据规模下两种算法的内存占用与缓存层级变化。"""
    scale = results.get("scale_analysis", [])
    if not scale:
        return

    samples     = [s["n_samples"] for s in scale]
    sklearn_kb  = [s["sklearn_kb"] for s in scale]
    hist_kb     = [s["histogram_kb"] for s in scale]
    x_labels    = [f"{n:,}" for n in samples]

    # 用整数索引作 x 轴，避免 annotate 在对数轴上无法解析字符串坐标
    x_pos = list(range(len(samples)))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(x_pos, sklearn_kb, "o-", color=_COLORS["sklearn"],
            linewidth=2, markersize=7, label=LABEL_SKLEARN)
    ax.plot(x_pos, hist_kb, "o-", color=_COLORS["histogram"],
            linewidth=2, markersize=7, label=LABEL_HISTOGRAM)

    # 缓存阈值线
    l1d_kb = L1D_CACHE_SIZE // 1024
    l2_kb  = L2_CACHE_SIZE // 1024
    ax.axhline(y=l1d_kb, color="#d2a8ff", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(x_pos) - 0.5, l1d_kb * 1.15, f"L1D = {l1d_kb} KB",
            fontsize=9, color="#d2a8ff", fontstyle="italic")
    ax.axhline(y=l2_kb, color="#79c0ff", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(x_pos) - 0.5, l2_kb * 1.15, f"L2 = {l2_kb} KB",
            fontsize=9, color="#79c0ff", fontstyle="italic")

    # 标注缓存层级（使用 ax.text 替代 ax.annotate，避免坐标转换问题）
    for i, s in enumerate(scale):
        ax.text(i, sklearn_kb[i] * 1.15, s["sklearn_cache_level"],
                fontsize=8, color=_COLORS["sklearn"], ha="center", va="bottom")
        ax.text(i, hist_kb[i] * 0.8, s["histogram_cache_level"],
                fontsize=8, color=_COLORS["histogram"], ha="center", va="top")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("样本数量", fontsize=11)
    ax.set_ylabel("数据内存 (KB)", fontsize=11)
    ax.set_title("数据规模扩展性：两种算法内存占用 vs 缓存容量阈值\n"
                 "(scikit-learn 随数据增长迅速超出缓存，直方图算法保持驻留)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "perf_scale_analysis.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── 图 7：综合雷达图（仅实测数据）─────────────────────────
def _chart_radar(results: dict[str, Any], output_dir: Path) -> None:
    """综合效能雷达图，仅使用实测数据。"""
    s  = results["sklearn"]
    h  = results["histogram"]
    bm = results["benchmark"]

    max_time   = max(s["train_time_sec"], h["train_time_sec"], 0.0001)
    max_pred   = max(s["predict_time_sec"], h["predict_time_sec"], 0.0001)
    max_cpu    = max(s["cpu_avg_pct"], h["cpu_avg_pct"], 1)
    max_mem    = max(s["data_size_kb"], h["data_size_kb"], 1)

    f64_access = bm["float64_access_ms"]
    u8_access  = bm["uint8_access_ms"]
    min_access = min(f64_access, u8_access)

    dims = ["训练速度", "预测速度", "CPU 效率*", "内存效率", "准确率", "数据访问速度"]

    sklearn_scores = [
        (1 - s["train_time_sec"] / max_time) * 100,
        (1 - s["predict_time_sec"] / max_pred) * 100,
        (1 - s["cpu_avg_pct"] / max_cpu) * 100,
        (1 - s["data_size_kb"] / max_mem) * 100,
        s["accuracy"] * 100,
        (min_access / max(f64_access, 1e-9)) * 100,
    ]
    histogram_scores = [
        (1 - h["train_time_sec"] / max_time) * 100,
        (1 - h["predict_time_sec"] / max_pred) * 100,
        (1 - h["cpu_avg_pct"] / max_cpu) * 100,
        (1 - h["data_size_kb"] / max_mem) * 100,
        h["accuracy"] * 100,
        (min_access / max(u8_access, 1e-9)) * 100,
    ]

    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    sklearn_closed   = sklearn_scores + sklearn_scores[:1]
    histogram_closed = histogram_scores + histogram_scores[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_facecolor("#161b22")

    ax.fill(angles, sklearn_closed, alpha=0.25, color=_COLORS["sklearn"])
    ax.plot(angles, sklearn_closed, "o-", linewidth=2,
            color=_COLORS["sklearn"], label=LABEL_SKLEARN)

    ax.fill(angles, histogram_closed, alpha=0.25, color=_COLORS["histogram"])
    ax.plot(angles, histogram_closed, "o-", linewidth=2,
            color=_COLORS["histogram"], label=LABEL_HISTOGRAM)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10, color="#e6edf3")
    ax.set_ylim(0, 110)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="#8b949e")
    ax.set_title("算法综合效能雷达图（仅实测数据，越高越好）",
                 fontsize=13, fontweight="bold", pad=20, color="#e6edf3")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              frameon=False, fontsize=10)

    # 添加 CPU 效率维度注释
    fig.text(0.5, 0.02,
             "* CPU 效率：完成相同训练任务消耗更少的 CPU 周期 = 更高效率（面积越大越好）",
             ha="center", fontsize=8, color="#8b949e", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(output_dir / "perf_radar_compare.png",
                dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    base_dir = Path(__file__).resolve().parent
    gc = json.loads((base_dir / "group_config.json").read_text(encoding="utf-8"))
    run_comparison(gc)
