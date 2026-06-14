"""Streamlit web interface for anomaly detection.

Features:
- Direct function imports (no subprocess)
- Model caching with st.cache_resource
- Data caching with st.cache_data
- Progress bar during execution
- Optional Plotly interactive charts
- Sidebar parameter controls
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# Import directly from modules (P0: no subprocess)
from data_generator import generate_dynamic_cases, generate_training_rows
from config import LABELS
from model import (
    format_run_summary,
    load_model,
    make_run_summary,
    predict_cases,
    save_model,
    train_model,
)
from utils import logger
from visualization import (
    PLOTLY_AVAILABLE,
    render_feature_importance_chart,
    render_stacked_bar_chart,
    render_status_chart,
    render_time_series_chart,
)

if PLOTLY_AVAILABLE:
    from visualization import render_interactive_pie, render_interactive_timeline

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "model.joblib"
PERF_DIR = OUTPUT_DIR / "perf_compare"


def read_json(path: Path) -> object | None:
    """Read JSON file safely."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {path}: {e}")
        return None


def load_config() -> tuple[dict | None, list | None]:
    """Load group config (cases generated dynamically at runtime)."""
    group_config = read_json(BASE_DIR / "group_config.json")
    return group_config, None


@st.cache_resource
def get_cached_model(model_path: Path):
    """Cache loaded model (P2: caching)."""
    return load_model(model_path)


@st.cache_data
def get_cached_predictions(predictions_path: Path):
    """Cache predictions (P2: caching)."""
    return read_json(predictions_path)


def run_pipeline(
    group_config: dict, force_retrain: bool = False
) -> tuple[bool, str]:
    """Run the full pipeline with progress bar (P0+P2)."""
    logs = []

    progress_bar = st.progress(0, text="初始化...")

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state["_current_run_signature"] = {
            "group_id": group_config.get("group_id"),
            "samples_per_label": group_config.get("samples_per_label"),
            "rf_params": dict(group_config.get("rf_params", {})),
        }

        # Step 1: Check for cached model
        progress_bar.progress(10, text="检查缓存模型...")
        model = None
        if not force_retrain and MODEL_PATH.exists():
            model = get_cached_model(MODEL_PATH)
            if model:
                logs.append("✓ 已加载缓存模型")

        # Step 2: Generate training data if needed
        progress_bar.progress(25, text="生成训练数据...")
        training_rows = generate_training_rows(group_config)
        logs.append(f"✓ 生成 {len(training_rows)} 条训练样本")

        # Step 3: Train model if cache unavailable
        if model is None:
            progress_bar.progress(50, text="训练模型...")
            model, metrics = train_model(training_rows, group_config)
            logs.append(f"✓ 模型训练完成，准确率: {metrics['accuracy']:.4f}")

            # Step 4: Save model
            progress_bar.progress(65, text="保存模型...")
            save_model(model, MODEL_PATH)
            (OUTPUT_DIR / "model_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            st.cache_resource.clear()
            logs.append("✓ 模型已保存")
        else:
            metrics = read_json(OUTPUT_DIR / "model_metrics.json") or {}
            logs.append("✓ 已复用缓存模型用于推理")

        # Step 5: Dynamic prediction
        progress_bar.progress(80, text="动态生成预测用例...")
        input_cases = generate_dynamic_cases(group_config)
        if not input_cases:
            raise ValueError("未生成任何预测用例，请检查 case_blueprints")
        predicted_rows = predict_cases(model, input_cases)
        logs.append(f"✓ 完成 {len(predicted_rows)} 条预测")

        # Step 6: Save outputs
        progress_bar.progress(90, text="保存结果...")
        summary = make_run_summary(
            group_config, training_rows, input_cases, predicted_rows, metrics
        )
        (OUTPUT_DIR / "predictions.json").write_text(
            json.dumps(predicted_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUTPUT_DIR / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUTPUT_DIR / "run_summary.txt").write_text(
            format_run_summary(summary), encoding="utf-8"
        )

        # Step 7: Render charts
        progress_bar.progress(95, text="渲染图表...")
        render_status_chart(predicted_rows, OUTPUT_DIR)
        render_time_series_chart(predicted_rows, OUTPUT_DIR)
        render_stacked_bar_chart(predicted_rows, OUTPUT_DIR)
        render_feature_importance_chart(model, OUTPUT_DIR)
        logs.append("✓ 图表渲染完成")

        progress_bar.progress(100, text="完成!")
        return True, "\n".join(logs)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        progress_bar.empty()
        return False, f"错误: {str(e)}\n{chr(10).join(logs)}"


# ========== Streamlit UI ==========

st.set_page_config(page_title="CANary - 金丝雀", layout="wide", page_icon="🛡️")
st.title("🛡️ CANary — 金丝雀异常检测可视化系统")

# Sidebar for parameter controls (P2)
with st.sidebar:
    st.header("⚙️ 参数控制")
    group_config, input_cases = load_config()

    if group_config:
        st.subheader("随机森林参数")
        rf_params = group_config.get("rf_params", {})
        n_estimators = st.slider(
            "树数量 (n_estimators)",
            50,
            300,
            int(rf_params.get("n_estimators", 100)),
        )
        max_depth = st.slider(
            "最大深度 (max_depth)", 3, 15, int(rf_params.get("max_depth", 7))
        )

        st.subheader("数据量参数")
        n_labels = len(LABELS)
        samples_per_label = st.slider(
            "每标签样本数 (samples_per_label)",
            50, 10000,
            int(group_config.get("samples_per_label", 320)),
            step=50,
            help="总样本量 = 每标签样本数 × 标签数（当前 6 个攻击类别）",
        )
        total_samples = samples_per_label * n_labels
        if total_samples <= 5000:
            size_hint = "小数据量"
        elif total_samples <= 30000:
            size_hint = "中数据量"
        else:
            size_hint = "大数据量"
        st.caption(f"预估总样本量：{total_samples:,}（{size_hint}）")

        if st.button("应用参数并重训练", type="secondary"):
            group_config["rf_params"]["n_estimators"] = n_estimators
            group_config["rf_params"]["max_depth"] = max_depth
            group_config["samples_per_label"] = samples_per_label
            (BASE_DIR / "group_config.json").write_text(
                json.dumps(group_config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # 删除缓存模型文件，强制下次用新参数重训练
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
            # 清除旧性能对比数据（参数变了，旧对比结果已失效）
            import glob as _glob
            for _stale_file in [
                PERF_DIR / "comparison_results.json",
                PERF_DIR / "perf_analysis.txt",
                PERF_DIR / "算法效能对比报告.pdf",
            ]:
                if _stale_file.exists():
                    _stale_file.unlink()
            # 同时清除旧图表文件
            for _chart in _glob.glob(str(PERF_DIR / "perf_*.png")):
                Path(_chart).unlink()
            # 清除旧 PDF 缓存
            st.session_state.pop("_pdf_report_bytes", None)
            st.cache_resource.clear()
            st.cache_data.clear()
            # 设置 session_state 标记，触发 rerun 后自动执行 pipeline
            st.session_state["_auto_retrain"] = True
            st.rerun()

# Main content
if group_config is None:
    st.error("无法加载配置文件，请检查 group_config.json")
    st.stop()

# ─── Tab 布局 ───
tab_train, tab_perf = st.tabs(["📊 训练结果", "⚡ 性能对比"])

# ======================== Tab 1: 训练结果 ========================
with tab_train:
    col1, col2 = st.columns([3, 1])
    with col1:
        run_clicked = st.button("🚀 运行 / 刷新结果", type="primary", use_container_width=True)
    with col2:
        force_retrain = st.checkbox("强制重训练", help="忽略缓存模型，使用当前参数重新训练（调整参数请用左侧“应用参数并重训练”按钮）")

    if run_clicked:
        ok, logs = run_pipeline(group_config, force_retrain=force_retrain)
        if ok:
            st.success("✅ 运行完成，结果已更新。")
            st.cache_data.clear()
        else:
            st.error("❌ 运行失败，请检查日志。")
        if logs:
            with st.expander("📋 运行日志", expanded=False):
                st.text_area("logs", logs, height=200, label_visibility="collapsed")

    # 参数重训练按钮触发的自动执行（仅触发一次）
    elif st.session_state.pop("_auto_retrain", False):
        with st.spinner("正在使用新参数重新训练模型..."):
            ok, logs = run_pipeline(group_config, force_retrain=True)
            if ok:
                st.success("✅ 新参数重训练完成，结果已更新。")
                st.cache_data.clear()
            else:
                st.error(f"❌ 重训练失败：{logs}")
            if logs:
                with st.expander("📋 重训练日志", expanded=True):
                    st.text_area("logs", logs, height=200, label_visibility="collapsed")
            st.rerun()  # 重训练后刷新页面展示最新结果

    # Display results
    summary_txt = OUTPUT_DIR / "run_summary.txt"
    if summary_txt.exists():
        st.subheader("📊 运行摘要")
        st.code(summary_txt.read_text(encoding="utf-8"))
    else:
        st.info('👈 请先点击「运行 / 刷新结果」按钮。')

    # Predictions table
    predictions = get_cached_predictions(OUTPUT_DIR / "predictions.json")
    if isinstance(predictions, list) and predictions:
        st.subheader("📋 预测结果")
        st.dataframe(predictions, use_container_width=True, height=400)

    # Charts section
    st.subheader("📈 可视化图表")

    if PLOTLY_AVAILABLE:
        chart_type = st.radio(
            "图表类型",
            ["静态图片 (Matplotlib)", "交互式图表 (Plotly)"],
            horizontal=True,
        )
        use_plotly = chart_type == "交互式图表 (Plotly)"
    else:
        use_plotly = False

    if use_plotly and PLOTLY_AVAILABLE and predictions:
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = render_interactive_pie(predictions)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_timeline = render_interactive_timeline(predictions)
            st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        col1, col2, col3 = st.columns(3)
        charts = [
            ("状态环图", OUTPUT_DIR / "status_donut.png", col1),
            ("时间分布图", OUTPUT_DIR / "attack_timeline.png", col2),
            ("等级堆叠图", OUTPUT_DIR / "attack_type_stacked.png", col3),
        ]
        for title, path, col in charts:
            with col:
                st.caption(title)
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.write("暂无图像")

    # Feature importance
    feature_importance_path = OUTPUT_DIR / "feature_importance.png"
    if feature_importance_path.exists():
        st.subheader("🔍 特征重要性")
        st.image(str(feature_importance_path), use_container_width=True)


# ======================== Tab 2: 性能对比 ========================
with tab_perf:
    st.subheader("⚡ 算法效能对比：传统 scikit-learn vs 直方图离散化 vs NLP 文本分类")

    # Run comparison button
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        perf_clicked = st.button(
            "🔬 运行性能对比测试", type="primary", use_container_width=True
        )

    if perf_clicked:
        with st.spinner("正在运行性能对比测试（含 NLP 文本分类训练，约 30~60 秒）..."):
            try:
                import time as _perf_time
                _t_start = _perf_time.perf_counter()
                from perf_compare import PerfBenchmark, run_comparison
                PERF_DIR.mkdir(parents=True, exist_ok=True)
                results = run_comparison(group_config, PERF_DIR)
                _elapsed = _perf_time.perf_counter() - _t_start
                # Save comparison results
                (PERF_DIR / "comparison_results.json").write_text(
                    json.dumps(
                        {k: v for k, v in results.items()
                         if k != "controlled_variables"},
                        ensure_ascii=False, indent=2, default=str,
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
                    summary_text + "\n\n" + conclusion_text,
                    encoding="utf-8",
                )
                st.session_state.pop("_pdf_report_bytes", None)  # 旧 PDF 已失效
                st.session_state["_last_perf_config"] = {
                    "samples_per_label": group_config.get("samples_per_label"),
                    "n_estimators": group_config.get("rf_params", {}).get("n_estimators"),
                    "max_depth": group_config.get("rf_params", {}).get("max_depth"),
                }
                st.success(f"✅ 性能对比测试完成！（耗时 {_elapsed:.1f} 秒）")
            except Exception as e:
                st.error(f"❌ 性能对比失败: {e}")
                logger.error(f"Perf comparison failed: {e}", exc_info=True)

    # ─── 显示已有对比结果 ───
    comparison_json = PERF_DIR / "comparison_results.json"
    if comparison_json.exists():
        comparison = read_json(comparison_json)
        if isinstance(comparison, dict):
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
            c = comparison.get("sklearn", {})
            e = comparison.get("histogram", {})
            mem = comparison.get("memory_analysis", {})
            config = comparison.get("config", {})
            hw_info = comparison.get("hardware_info", {})
            src_labels = comparison.get("source_labels", {})

            # ── 硬件配置环境说明 ──
            if hw_info:
                with st.expander("🖥️ 硬件配置环境", expanded=False):
                    hc1, hc2, hc3 = st.columns(3)
                    with hc1:
                        st.caption("**CPU 信息**")
                        cpu_model = hw_info.get("cpu_model", "Unknown")
                        # 截断过长的 CPU 型号名
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

            # ── 数据来源说明 ──
            if src_labels:
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

            # ── 区域 A：模型性能（随参数变化）──
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
                st.metric("直方图算法准确率", f"{e.get('accuracy', 0):.4f}",
                          delta=f"{e.get('accuracy', 0) - c.get('accuracy', 0):+.4f}")
            with mc2:
                st.metric("训练耗时 (scikit-learn)", f"{c.get('train_time_sec', 0):.4f}s")
                st.metric("训练耗时 (直方图)", f"{e.get('train_time_sec', 0):.4f}s")
            with mc3:
                st.metric("预测耗时 (scikit-learn)", f"{c.get('predict_time_sec', 0):.4f}s")
                st.metric("预测耗时 (直方图)", f"{e.get('predict_time_sec', 0):.4f}s")

            # ── 区域 A+：NLP-Transformer 模型（如果存在）──
            nlp_data = comparison.get("nlp", {})
            if nlp_data and nlp_data.get("accuracy"):
                st.markdown("---")
                st.markdown(
                    "#### :memo: NLP 文本分类模型（CAN 消息序列 token 化）"
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
                        delta=f"{nlp_data['accuracy'] - c.get('accuracy', 0):+.4f}"
                        f" vs sklearn",
                    )
                with nc2:
                    st.metric(
                        "训练耗时",
                        f"{nlp_data.get('train_time_sec', 0):.2f}s",
                    )
                with nc3:
                    st.metric(
                        "词表大小",
                        f"{nlp_data.get('vocab_size', 0):,}",
                    )
                with nc4:
                    nlp_cv = nlp_data.get("cv_mean_accuracy", 0)
                    st.metric(
                        "5-折 CV 准确率",
                        f"{nlp_cv:.4f}"
                        + (f" ± {nlp_data.get('cv_std_accuracy', 0):.4f}"
                           if nlp_cv else ""),
                    )
                st.caption(
                    f"模型: {nlp_data.get('model_name', 'logreg')} | "
                    f"特征维度: {nlp_data.get('n_features', '?')} | "
                    f"F1: {nlp_data.get('f1_macro', 0):.4f} | "
                    f"GPU 预留: {'是' if nlp_data.get('gpu_available') else '否'}"
                )

            # 当 scikit-learn 训练更快时，显示原因解释
            sk_train = c.get("train_time_sec", 0)
            hi_train = e.get("train_time_sec", 0)
            if sk_train > 0 and sk_train < hi_train:
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

            st.markdown("---")
            # 模型相关的图表
            model_charts = [
                ("perf_timing_compare.png", "训练 / 预测耗时"),
                ("perf_cpu_compare.png", "CPU 利用率（更低 = 更高效）"),
                ("perf_per_core_cpu.png", "逐核 CPU 热力图"),
                ("perf_cache_hit_rate.png", "缓存命中率对比"),
                ("perf_radar_compare.png", "综合效能雷达"),
            ]
            # NLP 对比图
            if comparison.get("nlp"):
                model_charts.insert(
                    0, ("perf_nlp_compare.png", "NLP-Transformer vs 传统 ML 对比")
                )
            for i in range(0, len(model_charts), 2):
                cols = st.columns(2)
                for j, (fname, title) in enumerate(model_charts[i:i+2]):
                    fpath = PERF_DIR / fname
                    with cols[j]:
                        st.caption(title)
                        if fpath.exists():
                            st.image(str(fpath), use_container_width=True)
                        else:
                            st.write("暂无图像")

            # ── 缓存命中率指标（在热力图下方）──
            sk_cache = c.get("cache", {})
            hi_cache = e.get("cache", {})
            if sk_cache and hi_cache:
                has_vtune = sk_cache.get("measured", False)
                cache_src = src_labels.get("cache", "")
                src_label = ("Intel VTune PMC 硬件实测"
                             if has_vtune
                             else "理论模型 (min(C/D, 1))")
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
                    st.metric("L1D 命中率 (直方图)", f"{hi_l1:.1f}%",
                              delta=f"{hi_l1 - sk_l1:+.1f}pp")
                with cc2:
                    sk_l2 = sk_cache.get("l2_hit_rate", 0) * 100
                    hi_l2 = hi_cache.get("l2_hit_rate", 0) * 100
                    st.metric("L2 命中率 (scikit-learn)", f"{sk_l2:.1f}%")
                    st.metric("L2 命中率 (直方图)", f"{hi_l2:.1f}%",
                              delta=f"{hi_l2 - sk_l2:+.1f}pp")
                with cc3:
                    sk_l3 = sk_cache.get("l3_hit_rate", 0) * 100
                    hi_l3 = hi_cache.get("l3_hit_rate", 0) * 100
                    st.metric("L3 命中率 (scikit-learn)", f"{sk_l3:.1f}%")
                    st.metric("L3 命中率 (直方图)", f"{hi_l3:.1f}%",
                              delta=f"{hi_l3 - sk_l3:+.1f}pp")
                with cc4:
                    sk_ov = sk_cache.get("effective_hit_rate", 0) * 100
                    hi_ov = hi_cache.get("effective_hit_rate", 0) * 100
                    st.metric("加权有效率 (scikit-learn)", f"{sk_ov:.2f}%")
                    st.metric("加权有效率 (直方图)", f"{hi_ov:.2f}%",
                              delta=f"{hi_ov - sk_ov:+.2f}pp")
                # VTune 实测事件计数
                if has_vtune:
                    sk_l3m = sk_cache.get("l3_miss_count", 0)
                    hi_l3m = hi_cache.get("l3_miss_count", 0)
                    sk_l2m = sk_cache.get("l2_miss_count", 0)
                    hi_l2m = hi_cache.get("l2_miss_count", 0)
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric("L2 Miss 计数 (scikit-learn)", f"{sk_l2m:,}")
                        st.metric("L2 Miss 计数 (直方图)", f"{hi_l2m:,}",
                                  delta=f"-{sk_l2m - hi_l2m:,}",
                                  delta_color="normal"
                                  if hi_l2m < sk_l2m else "inverse")
                    with mc2:
                        st.metric("L3 Miss 计数 (scikit-learn)", f"{sk_l3m:,}")
                        st.metric("L3 Miss 计数 (直方图)", f"{hi_l3m:,}",
                                  delta=f"-{sk_l3m - hi_l3m:,}",
                                  delta_color="normal"
                                  if hi_l3m < sk_l3m else "inverse")
                if hi_ov > sk_ov:
                    st.caption(
                        f"✅ 直方图算法整体缓存命中率高 **{hi_ov - sk_ov:.2f}pp**"
                        f" —— uint8 数据量仅为 float64 的 1/8，"
                        f"L3 Miss 更少意味着更少的 DRAM 访问，"
                        f"这是直方图算法性能优势的硬件级根因。"
                    )

            # ── 区域 B：数据层基准（仅取决于数据量）──
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
                st.metric("内存 (uint8)", f"{e.get('data_size_kb', 0):.1f} KB",
                          delta=f"-{mem.get('memory_reduction_pct', 0)}%")
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
            # 数据层基准相关的图表
            bench_charts = [
                ("perf_access_time.png", "多规模数据随机访问耗时"),
                ("perf_memory_compare.png", "数据集内存占用（确定性计算）"),
                ("perf_controlled_var.png", "控制变量隔离分析"),
                ("perf_scale_analysis.png", "数据规模扩展性"),
            ]
            for i in range(0, len(bench_charts), 2):
                cols = st.columns(2)
                for j, (fname, title) in enumerate(bench_charts[i:i+2]):
                    fpath = PERF_DIR / fname
                    with cols[j]:
                        st.caption(title)
                        if fpath.exists():
                            st.image(str(fpath), use_container_width=True)
                        else:
                            st.write("暂无图像")

            # 文本报告
            analysis_txt = PERF_DIR / "perf_analysis.txt"
            if analysis_txt.exists():
                report_col1, report_col2 = st.columns([3, 1])
                with report_col1:
                    with st.expander("📝 详细分析报告", expanded=False):
                        st.code(analysis_txt.read_text(encoding="utf-8"))
                with report_col2:
                    # PDF 报告下载
                    pdf_btn_key = "pdf_report_btn"
                    if st.button("📄 生成 PDF 报告", key=pdf_btn_key,
                                 use_container_width=True):
                        try:
                            from pdf_report import generate_pdf_report
                            from perf_compare import PerfBenchmark
                            conclusion_text = PerfBenchmark.generate_conclusion(comparison)
                            pdf_bytes = generate_pdf_report(
                                comparison, PERF_DIR, conclusion_text
                            )
                            # 同时保存到本地文件（备用）
                            pdf_path = PERF_DIR / "算法效能对比报告.pdf"
                            pdf_path.write_bytes(pdf_bytes)
                            st.session_state["_pdf_report_bytes"] = pdf_bytes
                            st.success(
                                f"✅ PDF 生成成功！({len(pdf_bytes)/1024:.0f} KB)\n\n"
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
    else:
        st.info("👈 请先点击「运行性能对比测试」按钮。")
