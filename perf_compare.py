"""算法效能对比 — 向后兼容重导出模块。

本模块保留原有 import 路径，实际实现已拆分到：
- perf_benchmark: PerfBenchmark 类 + 常量
- perf_chart: 图表渲染函数
- perf_analysis: run_comparison + 终端输出
"""

from perf_benchmark import (
    FLOAT64_SIZE,
    L1D_CACHE_LINE,
    L1D_CACHE_SIZE,
    L2_CACHE_SIZE,
    L3_CACHE_SIZE,
    LABEL_HISTOGRAM,
    LABEL_NLP,
    LABEL_SKLEARN,
    NUM_FEATURES,
    SHORT_HISTOGRAM,
    SHORT_NLP,
    SHORT_SKLEARN,
    UINT8_SIZE,
    PerfBenchmark,
)
from perf_chart import render_comparison_charts
from perf_analysis import run_comparison

__all__ = [
    "PerfBenchmark",
    "run_comparison",
    "render_comparison_charts",
    "L1D_CACHE_LINE",
    "L1D_CACHE_SIZE",
    "L2_CACHE_SIZE",
    "L3_CACHE_SIZE",
    "FLOAT64_SIZE",
    "UINT8_SIZE",
    "NUM_FEATURES",
    "LABEL_SKLEARN",
    "LABEL_HISTOGRAM",
    "LABEL_NLP",
    "SHORT_SKLEARN",
    "SHORT_HISTOGRAM",
    "SHORT_NLP",
]
