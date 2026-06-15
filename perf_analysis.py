"""算法效能对比 — 运行器与分析模块。

串联三种算法的端到端对比流程：
1. 生成共享训练数据
2. 分别运行 scikit-learn、直方图算法 和 NLP 文本分类器
3. 收集各维度效能指标
4. 输出终端报告和图表
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from perf_benchmark import (
    FLOAT64_SIZE,
    LABEL_HISTOGRAM,
    LABEL_NLP,
    LABEL_SKLEARN,
    NUM_FEATURES,
    UINT8_SIZE,
    PerfBenchmark,
)
from perf_chart import render_comparison_charts
from data_generator import generate_dynamic_cases, generate_training_rows
from histogram_model import (
    predict_histogram_cases,
    train_histogram_model,
)
from model import (
    predict_cases,
    train_model,
)
from utils import logger

# NLP 模块（可选，PyTorch 未安装时跳过）
try:
    from nlp_data import generate_can_sequences, prepare_nlp_dataset
    from nlp_model import train_nlp_model, predict_nlp_sequences
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


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

    # ── NLP 文本分类算法 ────────────────────────────
    nlp_results: dict[str, Any] | None = None
    nlp_train_time = 0.0
    nlp_predict_time = 0.0
    if NLP_AVAILABLE:
        logger.info("\n" + "─" * 50)
        logger.info(f"【{LABEL_NLP}】CAN 消息序列 → 文本分类")
        logger.info("─" * 50)
        try:
            # NLP 样本数上限：避免 Web 端长时间卡死
            # NLP 仅用于概念对比演示，500/标签（共 3000 条）已足够
            samples_per_label_nlp = min(
                int(group_config.get("samples_per_label", 320)), 500
            )
            seq_len = min(64, max(16, samples_per_label_nlp * 6 // 100))
            logger.info(f"[NLP] 生成 CAN 文本样本 (seq_len={seq_len}, "
                       f"samples_per_label={samples_per_label_nlp})...")
            # 临时缩小 samples_per_label 以加速 NLP 生成
            nlp_config = dict(group_config)
            nlp_config["samples_per_label"] = samples_per_label_nlp
            nlp_sequences = generate_can_sequences(nlp_config, seq_len=seq_len)
            nlp_dataset = prepare_nlp_dataset(nlp_sequences, seq_len=seq_len)
            logger.info(f"[NLP] 生成 {nlp_dataset['n_samples']} 条文本样本")

            t0 = time.perf_counter()
            nlp_model, nlp_metrics = train_nlp_model(nlp_dataset, group_config)
            nlp_train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            nlp_predictions = predict_nlp_sequences(nlp_model, nlp_dataset)
            nlp_predict_time = time.perf_counter() - t0

            nlp_results = {
                "accuracy": nlp_metrics["accuracy"],
                "train_time_sec": round(nlp_train_time, 4),
                "predict_time_sec": round(nlp_predict_time, 4),
                "cv_mean_accuracy": nlp_metrics.get("cv_mean_accuracy", 0),
                "cv_std_accuracy": nlp_metrics.get("cv_std_accuracy", 0),
                "vocab_size": nlp_metrics.get("vocab_size", 0),
                "seq_len": nlp_metrics.get("seq_len", seq_len),
                "model_name": nlp_metrics.get("model_name", "logreg"),
                "gpu_available": nlp_metrics.get("gpu_available", False),
                "n_features": nlp_metrics.get("n_features", 0),
                "f1_macro": nlp_metrics.get("f1_macro", 0),
                "label": LABEL_NLP,
                "predictions_preview": nlp_predictions[:3],
            }
            logger.info(f"[NLP] 训练完成: Acc={nlp_metrics['accuracy']:.4f}, "
                       f"Time={nlp_train_time:.2f}s")
        except Exception as _nlp_err:
            logger.warning(f"[NLP] 训练失败 (将继续使用 RF 对比): {_nlp_err}")
            nlp_results = None
    else:
        logger.info("\n[NLP] 依赖不可用，跳过 NLP 文本分类对比")

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
    source_labels = PerfBenchmark.get_source_labels(cache)
    hardware_info = PerfBenchmark.get_hardware_info()
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
        "hardware_info": hardware_info,
        "source_labels": source_labels,
    }

    # ── 附上 NLP 结果（如果存在）──
    if nlp_results:
        nlp_results["source"] = "GPU 预留 token 化 + CPU 文本分类"
        results["nlp"] = nlp_results

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
    print(f"  算法效能对比报告：{LABEL_SKLEARN} vs {LABEL_HISTOGRAM}"
          + (f" vs {LABEL_NLP}" if results.get("nlp") else ""))
    print("█" * 64)

    nlp = results.get("nlp", {})
    has_nlp = bool(nlp)

    print("\n── 模型准确率 ──")
    print(f"  {LABEL_SKLEARN}:    {s['accuracy']:.4f}")
    print(f"  {LABEL_HISTOGRAM}:  {h['accuracy']:.4f}")
    if has_nlp:
        print(f"  {LABEL_NLP}: {nlp['accuracy']:.4f}")
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

    if has_nlp:
        print("\n── NLP 文本分类训练耗时 ──")
        print(f"  {LABEL_NLP}: {nlp['train_time_sec']:.4f}s")
        print(f"  词表大小:   {nlp.get('vocab_size', 0):,}")
        print(f"  模型:       {nlp.get('model_name', 'logreg')}")
        if nlp.get('cv_mean_accuracy'):
            print(f"  5-折CV:     {nlp['cv_mean_accuracy']:.4f}"
                  f" ± {nlp['cv_std_accuracy']:.4f}")

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
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json
    base_dir = Path(__file__).resolve().parent
    gc = _json.loads((base_dir / "group_config.json").read_text(encoding="utf-8"))
    run_comparison(gc)
