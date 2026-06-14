"""算法效能对比 — 图表渲染模块。

生成多维度算法效能对比图：
- 多规模数据访问耗时对比
- CPU 利用率 & 逐核热力图
- 缓存命中率对比
- 训练/预测耗时对比
- 内存占用对比
- 控制变量隔离分析
- 数据规模扩展性分析
- 综合效能雷达图
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perf_benchmark import (
    L1D_CACHE_SIZE,
    L2_CACHE_SIZE,
    LABEL_HISTOGRAM,
    LABEL_SKLEARN,
    SHORT_HISTOGRAM,
    SHORT_SKLEARN,
)
from utils import logger


# ---------------------------------------------------------------------------
# matplotlib 配置
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


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

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
    ax1.bar([x - width / 2 for x in x_pos], sk_l1, width,
            color=_COLORS["sklearn"], label=LABEL_SKLEARN,
            edgecolor="white", linewidth=0.6)
    ax1.bar([x + width / 2 for x in x_pos], hi_l1, width,
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
    ax2.bar([x - width / 2 for x in x_pos], sk_l3, width,
            color=_COLORS["sklearn"], label=LABEL_SKLEARN,
            edgecolor="white", linewidth=0.6)
    ax2.bar([x + width / 2 for x in x_pos], hi_l3, width,
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

    bar_s = ax.bar(x - width / 2, [s["train_time_sec"], h["train_time_sec"]], width,
                   color=[_COLORS["sklearn"], _COLORS["histogram"]],
                   edgecolor="white", linewidth=0.6, label="训练耗时")
    bar_h = ax.bar(x + width / 2, [s["predict_time_sec"], h["predict_time_sec"]], width,
                   color=[_COLORS["sklearn_light"], _COLORS["histogram_light"]],
                   edgecolor="white", linewidth=0.6, label="预测耗时", alpha=0.75)

    for bar in [*bar_s, *bar_h]:
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
    x = np.arange(1)

    ax.bar(x - 0.15, [s["data_size_kb"]], 0.3,
           color=_COLORS["sklearn"], edgecolor="white", linewidth=0.6,
           label=LABEL_SKLEARN)
    ax.bar(x + 0.15, [h["data_size_kb"]], 0.3,
           color=_COLORS["histogram"], edgecolor="white", linewidth=0.6,
           label=LABEL_HISTOGRAM)

    for i, val in enumerate([s["data_size_kb"]]):
        ax.text(i - 0.15, val + max(s["data_size_kb"], 1) * 0.01,
                f"{val:.0f} KB", ha="center", va="bottom", fontsize=10, color="#e6edf3")
    for i, val in enumerate([h["data_size_kb"]]):
        ax.text(i + 0.15, val + max(h["data_size_kb"], 1) * 0.01,
                f"{val:.0f} KB", ha="center", va="bottom", fontsize=10, color="#e6edf3")

    l1d_kb = L1D_CACHE_SIZE // 1024
    ax.axhline(y=l1d_kb, color="#d2a8ff", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(0.45, l1d_kb + max(s["data_size_kb"], 1) * 0.015,
            f"L1D Cache = {l1d_kb} KB", fontsize=9, color="#d2a8ff", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(["数据集内存占用"], fontsize=11)
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
    ax.bar(["C-order", "F-order"],
           [d_f64["c_order_ms"], d_f64["f_order_ms"]],
           color=[_COLORS["histogram"], _COLORS["sklearn"]],
           width=0.4, edgecolor="white", linewidth=0.6)
    for bar, v in zip(ax.patches, [d_f64["c_order_ms"], d_f64["f_order_ms"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", fontsize=9, color="#e6edf3")
    ax.set_title(f"固定 float64（scikit-learn 的数据类型）\nF-order 慢 {d_f64['slowdown_f_vs_c']}×",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("ms")

    # 维度 2：固定 C-order，改 dtype
    ax = axes[1]
    ax.bar(["float64\n(scikit-learn)", "uint8\n(直方图)"],
           [d_co["float64_ms"], d_co["uint8_ms"]],
           color=[_COLORS["sklearn"], _COLORS["histogram"]],
           width=0.4, edgecolor="white", linewidth=0.6)
    for bar, v in zip(ax.patches, [d_co["float64_ms"], d_co["uint8_ms"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", fontsize=9, color="#e6edf3")
    ax.set_title(f"固定 C-order（相同内存布局）\nuint8 快 {d_co['speedup_u8_vs_f64']}×",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("ms")

    # 维度 3：固定 uint8，改 order
    ax = axes[2]
    ax.bar(["C-order", "F-order"],
           [d_u8["c_order_ms"], d_u8["f_order_ms"]],
           color=[_COLORS["histogram"], _COLORS["sklearn"]],
           width=0.4, edgecolor="white", linewidth=0.6)
    for bar, v in zip(ax.patches, [d_u8["c_order_ms"], d_u8["f_order_ms"]]):
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
