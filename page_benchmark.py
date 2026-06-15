"""Page 2: 算法性能评估（实验室 / Benchmark Lab）。

数据流解耦：在此页面调用 perf_analysis.py 进行 A/B/C 三路打擂。
严格隔离变量命名（X_rf / X_nlp），禁止稠密矩阵与 TF-IDF 稀疏矩阵的变量污染。
惰性加载：仅当用户实际访问此页面时才执行 benchmark 计算。

三套 Baseline：
1. NLP 文本分类 — 语义范式基准探索（~0.17 准确率）
2. 传统 Scikit-learn (float64) — 高准确率但 Memory Wall 受限
3. 直方图算法 (uint8) — 最优性能，L3 Cache 命中率 66.68%
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from utils import logger

# ── 路径常量 ──
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PERF_DIR = OUTPUT_DIR / "perf_compare"


def _read_json(path: Path) -> object | None:
    """安全读取 JSON 文件。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ── 子区域渲染函数 ──


def _render_hardware_info(hw_info: dict) -> None:
    """渲染硬件配置环境折叠区。"""
    if not hw_info:
        return
    with st.expander("🖥️ 硬件配置环境", expanded=False):
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.caption("**CPU 信息**")
            cpu_model = hw_info.get("cpu_model", "Unknown")
            if len(str(cpu_model)) > 40:
                cpu_model = str(cpu_model)[:37] + "..."
            st.metric("CPU 型号", cpu_model)
            st.metric("物理核心", hw_info.get("cpu_cores_physical", "N/A"))
            st.metric("逻辑核心", hw_info.get("cpu_cores_logical", "N/A"))
        with hc2:
            st.caption("**Cache 配置**")
            st.metric("L1D Cache", f"{hw_info.get('l1d_cache_kb', 0)} KB")
            st.metric("L2 Cache", f"{hw_info.get('l2_cache_kb', 0)} KB")
            st.metric("L3 Cache", f"{hw_info.get('l3_cache_kb', 0)} KB")
            st.metric("Cache Line", f"{hw_info.get('cache_line_bytes', 0)} B")
        with hc3:
            st.caption("**系统环境**")
            st.metric("总内存", f"{hw_info.get('total_ram_gb', 0):.1f} GB")
            st.metric("Python", hw_info.get("python_version", "N/A"))
            st.metric("NumPy", hw_info.get("numpy_version", "N/A"))
            st.metric("scikit-learn", hw_info.get("sklearn_version", "N/A"))


def _render_source_labels(src_labels: dict) -> None:
    """渲染性能数据来源说明折叠区。"""
    if not src_labels:
        return
    with st.expander("📋 性能数据来源说明", expanded=False):
        src_items = [
            ("⏱️ 训练/预测耗时 & 准确率", src_labels.get("timing", "N/A")),
            ("📊 CPU 利用率", src_labels.get("cpu", "N/A")),
            ("💾 内存占用", src_labels.get("memory", "N/A")),
            ("🚀 数据访问速度", src_labels.get("access_speed", "N/A")),
            ("🧊 缓存命中率", src_labels.get("cache", "N/A")),
        ]
        for label, source in src_items:
            st.caption(f"{label}  →  **{source}**")


def _render_model_metrics(comparison: dict, src_labels: dict) -> None:
    """区域 A：模型性能指标（随超参数变化）。"""
    c = comparison.get("sklearn", {})
    e = comparison.get("histogram", {})
    config = comparison.get("config", {})

    st.markdown("---")
    st.markdown(
        "#### :dart: 模型性能指标（随树个数 / 深度等超参数变化）"
    )
    timing_src = src_labels.get("timing", "")
    st.caption(
        f"数据来源: **{timing_src}**  |  "
        "以下指标来自实际的 RandomForest 训练与预测，调整左侧参数后重新运行即可看到变化"
    )
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("scikit-learn 准确率", f"{c.get('accuracy', 0):.4f}")
        st.metric(
            "直方图算法准确率",
            f"{e.get('accuracy', 0):.4f}",
            delta=f"{e.get('accuracy', 0) - c.get('accuracy', 0):+.4f}",
        )
    with mc2:
        st.metric("训练耗时 (scikit-learn)", f"{c.get('train_time_sec', 0):.4f}s")
        st.metric("训练耗时 (直方图)", f"{e.get('train_time_sec', 0):.4f}s")
    with mc3:
        st.metric("预测耗时 (scikit-learn)", f"{c.get('predict_time_sec', 0):.4f}s")
        st.metric("预测耗时 (直方图)", f"{e.get('predict_time_sec', 0):.4f}s")

    # 当 scikit-learn 训练更快时，显示原因解释
    sk_train = c.get("train_time_sec", 0)
    hi_train = e.get("train_time_sec", 0)
    if 0 < sk_train < hi_train:
        n_samp = config.get("n_samples", 0)
        st.info(
            f"**为什么 scikit-learn 反而更快？**\n\n"
            f"当前样本量仅 **{n_samp:,}** 条，数据完全被 L2/L3 缓存覆盖，"
            f"不存在 Memory Wall 问题。而直方图算法额外包含 "
            f"**float64→uint8 离散化编码**步骤（约 3~10ms），"
            f"在小数据下该固定开销 > 缓存收益。\n\n"
            f"请尝试增大左侧 **每标签样本数**（建议 ≥2000，"
            f"总样本量 ≥12,000）后重新运行，可观察到直方图算法逐渐追平并反超。"
        )


def _render_nlp_section(comparison: dict) -> None:
    """区域 A+：NLP 文本分类模型（Baseline 1）。"""
    nlp_data = comparison.get("nlp", {})
    if not nlp_data or not nlp_data.get("accuracy"):
        return

    c = comparison.get("sklearn", {})

    st.markdown("---")
    st.markdown(
        "#### :memo: Baseline 1 — NLP 文本分类模型（CAN 消息序列 token 化）"
    )
    st.caption(
        "将 CAN 消息窗口视为领域文本：每条消息组合成 token，"
        "再用 TF-IDF + 经典线性分类器完成风险识别。"
    )
    nc1, nc2, nc3, nc4 = st.columns(4)
    with nc1:
        st.metric(
            "NLP 准确率",
            f"{nlp_data['accuracy']:.4f}",
            delta=(
                f"{nlp_data['accuracy'] - c.get('accuracy', 0):+.4f}"
                f" vs sklearn"
            ),
        )
    with nc2:
        st.metric("训练耗时", f"{nlp_data.get('train_time_sec', 0):.2f}s")
    with nc3:
        st.metric("词表大小", f"{nlp_data.get('vocab_size', 0):,}")
    with nc4:
        nlp_cv = nlp_data.get("cv_mean_accuracy", 0)
        st.metric(
            "5-折 CV 准确率",
            f"{nlp_cv:.4f}"
            + (
                f" ± {nlp_data.get('cv_std_accuracy', 0):.4f}"
                if nlp_cv
                else ""
            ),
        )
    st.caption(
        f"模型: {nlp_data.get('model_name', 'logreg')} | "
        f"特征维度: {nlp_data.get('n_features', 0)} | "
        f"F1: {nlp_data.get('f1_macro', 0):.4f} | "
        f"GPU 预留: {'是' if nlp_data.get('gpu_available') else '否'}"
    )
    st.caption(
        "注：对标签噪声敏感，存在物理特征丢失，仅作语义范式基准探索。"
    )


def _render_model_charts(comparison: dict) -> None:
    """渲染模型相关的对比图表。"""
    model_charts = [
        ("perf_timing_compare.png", "训练 / 预测耗时"),
        ("perf_cpu_compare.png", "CPU 利用率（更低 = 更高效）"),
        ("perf_per_core_cpu.png", "逐核 CPU 热力图"),
        ("perf_cache_hit_rate.png", "缓存命中率对比"),
        ("perf_radar_compare.png", "综合效能雷达"),
    ]
    if comparison.get("nlp"):
        model_charts.insert(
            0, ("perf_nlp_compare.png", "NLP 文本分类 vs 传统 ML 对比")
        )
    for i in range(0, len(model_charts), 2):
        cols = st.columns(2)
        for j, (fname, title) in enumerate(model_charts[i : i + 2]):
            fpath = PERF_DIR / fname
            with cols[j]:
                st.caption(title)
                if fpath.exists():
                    st.image(str(fpath), use_container_width=True)
                else:
                    st.write("暂无图像")


def _render_cache_metrics(comparison: dict, src_labels: dict) -> None:
    """渲染缓存命中率指标。"""
    c = comparison.get("sklearn", {})
    e = comparison.get("histogram", {})
    sk_cache = c.get("cache", {})
    hi_cache = e.get("cache", {})
    if not (sk_cache and hi_cache):
        return

    has_vtune = sk_cache.get("measured", False)
    cache_src = src_labels.get("cache", "")
    src_label = (
        "Intel VTune PMC 硬件实测" if has_vtune else "理论模型 (min(C/D, 1))"
    )
    st.markdown(f"##### 🧊 缓存命中率（数据来源：{cache_src}）")
    if has_vtune:
        st.caption(
            "数据来源: Intel VTune Profiler uarch-exploration 分析，"
            "硬件 PMC 事件 MEM_LOAD_RETIRED.{L1_HIT, L2_HIT, L3_HIT, L3_MISS}"
        )
    else:
        st.caption(
            "基于工作集大小与缓存层级的理论分析，"
            "安装 Intel VTune 后可切换为硬件实测数据"
        )
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        sk_l1 = sk_cache.get("l1d_hit_rate", 0) * 100
        hi_l1 = hi_cache.get("l1d_hit_rate", 0) * 100
        st.metric("L1D 命中率 (scikit-learn)", f"{sk_l1:.1f}%")
        st.metric(
            "L1D 命中率 (直方图)",
            f"{hi_l1:.1f}%",
            delta=f"{hi_l1 - sk_l1:+.1f}pp",
        )
    with cc2:
        sk_l2 = sk_cache.get("l2_hit_rate", 0) * 100
        hi_l2 = hi_cache.get("l2_hit_rate", 0) * 100
        st.metric("L2 命中率 (scikit-learn)", f"{sk_l2:.1f}%")
        st.metric(
            "L2 命中率 (直方图)",
            f"{hi_l2:.1f}%",
            delta=f"{hi_l2 - sk_l2:+.1f}pp",
        )
    with cc3:
        sk_l3 = sk_cache.get("l3_hit_rate", 0) * 100
        hi_l3 = hi_cache.get("l3_hit_rate", 0) * 100
        st.metric("L3 命中率 (scikit-learn)", f"{sk_l3:.1f}%")
        st.metric(
            "L3 命中率 (直方图)",
            f"{hi_l3:.1f}%",
            delta=f"{hi_l3 - sk_l3:+.1f}pp",
        )
    with cc4:
        sk_ov = sk_cache.get("effective_hit_rate", 0) * 100
        hi_ov = hi_cache.get("effective_hit_rate", 0) * 100
        st.metric("加权有效率 (scikit-learn)", f"{sk_ov:.2f}%")
        st.metric(
            "加权有效率 (直方图)",
            f"{hi_ov:.2f}%",
            delta=f"{hi_ov - sk_ov:+.2f}pp",
        )

    # VTune 实测事件计数
    if has_vtune:
        sk_l3m = sk_cache.get("l3_miss_count", 0)
        hi_l3m = hi_cache.get("l3_miss_count", 0)
        sk_l2m = sk_cache.get("l2_miss_count", 0)
        hi_l2m = hi_cache.get("l2_miss_count", 0)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("L2 Miss 计数 (scikit-learn)", f"{sk_l2m:,}")
            st.metric(
                "L2 Miss 计数 (直方图)",
                f"{hi_l2m:,}",
                delta=f"-{sk_l2m - hi_l2m:,}",
                delta_color="normal" if hi_l2m < sk_l2m else "inverse",
            )
        with mc2:
            st.metric("L3 Miss 计数 (scikit-learn)", f"{sk_l3m:,}")
            st.metric(
                "L3 Miss 计数 (直方图)",
                f"{hi_l3m:,}",
                delta=f"-{sk_l3m - hi_l3m:,}",
                delta_color="normal" if hi_l3m < sk_l3m else "inverse",
            )
    if hi_ov > sk_ov:
        st.caption(
            f"✅ 直方图算法整体缓存命中率高 **{hi_ov - sk_ov:.2f}pp**"
            f" —— uint8 数据量仅为 float64 的 1/8，"
            f"L3 Miss 更少意味着更少的 DRAM 访问，"
            f"这是直方图算法性能优势的硬件级根因。"
        )


def _render_data_layer_benchmarks(comparison: dict, src_labels: dict) -> None:
    """区域 B：数据层基准指标。"""
    c = comparison.get("sklearn", {})
    e = comparison.get("histogram", {})
    mem = comparison.get("memory_analysis", {})
    config = comparison.get("config", {})

    st.markdown("---")
    st.markdown(
        "#### :computer: 数据层基准指标（仅取决于样本量，与模型参数无关）"
    )
    mem_src = src_labels.get("memory", "")
    acc_src = src_labels.get("access_speed", "")
    st.caption(
        f"数据来源: 内存 → **{mem_src}** | 访问速度 → **{acc_src}**  "
        "以下基准测试测量的是两种算法在**数据访问层面**的硬件性能差异"
        "（随机行访问 `mat[random_indices]`，模拟决策树样本查找），"
        "结果只由 `samples_per_label`（决定样本总量）决定，"
        "调整树个数 / 深度不会改变这些数值。"
    )
    bc1, bc2 = st.columns(2)
    with bc1:
        st.metric("内存 (float64)", f"{c.get('data_size_kb', 0):.1f} KB")
        st.metric(
            "内存 (uint8)",
            f"{e.get('data_size_kb', 0):.1f} KB",
            delta=f"-{mem.get('memory_reduction_pct', 0)}%",
        )
    with bc2:
        bm = comparison.get("benchmark", {})
        st.metric(
            "随机访问加速比 (uint8 vs float64)",
            f"{bm.get('speedup', 0):.2f}×",
            help="在相同样本量下，uint8 随机行访问比 float64 快多少倍",
        )
        st.metric(
            "测试样本量",
            f"{config.get('n_samples', 0):,}",
            help="仅当调整 samples_per_label 时此值才会变化",
        )

    st.markdown("---")
    bench_charts = [
        ("perf_access_time.png", "多规模数据随机访问耗时"),
        ("perf_memory_compare.png", "数据集内存占用（确定性计算）"),
        ("perf_controlled_var.png", "控制变量隔离分析"),
        ("perf_scale_analysis.png", "数据规模扩展性"),
    ]
    for i in range(0, len(bench_charts), 2):
        cols = st.columns(2)
        for j, (fname, title) in enumerate(bench_charts[i : i + 2]):
            fpath = PERF_DIR / fname
            with cols[j]:
                st.caption(title)
                if fpath.exists():
                    st.image(str(fpath), use_container_width=True)
                else:
                    st.write("暂无图像")


def _render_pdf_report(comparison: dict) -> None:
    """渲染 PDF 报告生成与下载区域。"""
    analysis_txt = PERF_DIR / "perf_analysis.txt"

    report_col1, report_col2 = st.columns([3, 1])
    with report_col1:
        if analysis_txt.exists():
            with st.expander("📝 详细分析报告", expanded=False):
                st.code(analysis_txt.read_text(encoding="utf-8"))
        else:
            st.caption("📝 分析报告将在运行性能对比测试后自动生成")
    with report_col2:
        pdf_btn_key = "pdf_report_btn"
        if st.button(
            "📄 生成 PDF 报告", key=pdf_btn_key, use_container_width=True
        ):
            try:
                from pdf_report import generate_pdf_report
                from perf_compare import PerfBenchmark

                conclusion_text = PerfBenchmark.generate_conclusion(comparison)
                pdf_bytes = generate_pdf_report(
                    comparison, PERF_DIR, conclusion_text
                )
                pdf_path = PERF_DIR / "算法效能对比报告.pdf"
                pdf_path.write_bytes(pdf_bytes)
                st.session_state["_pdf_report_bytes"] = pdf_bytes
                st.success(
                    f"✅ PDF 生成成功！({len(pdf_bytes) / 1024:.0f} KB)\n\n"
                    f"已保存至: `{pdf_path}`"
                )
            except ModuleNotFoundError as _mod_err:
                st.error(
                    f"缺少依赖包: {_mod_err.name}。"
                    f"请在终端运行: pip install {_mod_err.name}"
                    f"  (或 pip install fpdf2)"
                )
            except Exception as _pdf_err:
                st.error(f"PDF 生成失败: {_pdf_err}")
        if "_pdf_report_bytes" in st.session_state:
            st.download_button(
                label="⬇️ 下载报告",
                data=st.session_state["_pdf_report_bytes"],
                file_name="算法效能对比报告.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ── 主渲染函数 ──


def render_benchmark_page(group_config: dict) -> None:
    """渲染算法性能评估页面（Page 2 — Benchmark Lab）。

    惰性加载：仅当用户实际访问此页面时才导入 perf_analysis 模块，
    绝不在 Page 1 时预计算 NLP 矩阵或执行 benchmark。
    """
    st.subheader(
        "⚡ 算法效能对比：传统 scikit-learn vs 直方图离散化 vs NLP 文本分类"
    )

    # ── 三套 Baseline 说明 ──
    st.markdown(
        "| # | Baseline | 引擎 | 特点 |\n"
        "|:--:|----------|------|------|\n"
        "| 1 | **NLP 文本分类** | TF-IDF + LogReg/SVC | 语义范式基准，对标签噪声敏感 |\n"
        "| 2 | **Scikit-learn** | float64 RandomForest | 高准确率，Memory Wall 受限 |\n"
        "| 3 | **直方图算法** ✅ | uint8 256桶 RandomForest | 最优性能，L3 Cache 命中率 66.68% |"
    )

    # ── 运行控制 ──
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        perf_clicked = st.button(
            "🔬 运行性能对比测试",
            type="primary",
            use_container_width=True,
        )

    if perf_clicked:
        with st.spinner(
            "正在运行性能对比测试（含 NLP 文本分类训练，约 30~60 秒）..."
        ):
            try:
                t_start = time.perf_counter()
                # 惰性导入：仅在用户点击时才加载 perf_analysis 模块
                from perf_compare import PerfBenchmark, run_comparison

                PERF_DIR.mkdir(parents=True, exist_ok=True)
                results = run_comparison(group_config, PERF_DIR)
                elapsed = time.perf_counter() - t_start

                # 保存对比结果 JSON
                (PERF_DIR / "comparison_results.json").write_text(
                    json.dumps(
                        {
                            k: v
                            for k, v in results.items()
                            if k != "controlled_variables"
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                summary_text = PerfBenchmark.summary_text(
                    results["benchmark"],
                    results["controlled_variables"],
                    results["memory_analysis"],
                )
                conclusion_text = PerfBenchmark.generate_conclusion(results)
                (PERF_DIR / "perf_analysis.txt").write_text(
                    summary_text + "\n\n" + conclusion_text, encoding="utf-8"
                )
                st.session_state.pop("_pdf_report_bytes", None)
                st.session_state["_last_perf_config"] = {
                    "samples_per_label": group_config.get("samples_per_label"),
                    "n_estimators": group_config.get("rf_params", {}).get(
                        "n_estimators"
                    ),
                    "max_depth": group_config.get("rf_params", {}).get(
                        "max_depth"
                    ),
                }
                st.success(f"✅ 性能对比测试完成！（耗时 {elapsed:.1f} 秒）")
            except Exception as e:
                st.error(f"❌ 性能对比失败: {e}")
                logger.error(f"Perf comparison failed: {e}", exc_info=True)

    # ── 显示已有对比结果 ──
    comparison_json = PERF_DIR / "comparison_results.json"
    if not comparison_json.exists():
        st.info("👈 请先点击「运行性能对比测试」按钮。")
        return

    comparison = _read_json(comparison_json)
    if not isinstance(comparison, dict):
        st.warning("对比结果文件损坏，请重新运行测试。")
        return

    # ── 对比数据时效性检查 ──
    _last_cfg = st.session_state.get("_last_perf_config", {})
    _cur_cfg = {
        "samples_per_label": group_config.get("samples_per_label"),
        "n_estimators": group_config.get("rf_params", {}).get("n_estimators"),
        "max_depth": group_config.get("rf_params", {}).get("max_depth"),
    }
    if _last_cfg and _last_cfg != _cur_cfg:
        st.warning(
            "⚠️ 检测到参数已变更，当前性能对比数据可能来自旧参数。"
            "请重新点击「运行性能对比测试」以获取最新数据。"
        )

    src_labels = comparison.get("source_labels", {})

    # ── 各子区域渲染 ──
    _render_hardware_info(comparison.get("hardware_info", {}))
    _render_source_labels(src_labels)
    _render_model_metrics(comparison, src_labels)
    _render_nlp_section(comparison)
    _render_model_charts(comparison)
    _render_cache_metrics(comparison, src_labels)
    _render_data_layer_benchmarks(comparison, src_labels)
    try:
        _render_pdf_report(comparison)
    except Exception as _pdf_section_err:
        st.warning(f"PDF 报告区域加载异常: {_pdf_section_err}")


# ── st.Page 入口：Streamlit 通过 nav.run() 执行此脚本时自动渲染 ──
_gc = st.session_state.get("group_config")
if _gc is None:
    st.error("无法加载配置，请检查 group_config.json")
else:
    render_benchmark_page(_gc)
