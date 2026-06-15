"""Page 1: 生产看板 — 直方图降维随机森林引擎。

唯一指定引擎：histogram_model.py（uint8 256 桶离散化 RandomForest）。
绝对禁止在此页面引入或执行任何 NLP 相关的 TF-IDF 或文本分类代码。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import PREDICTION_COLUMNS_CN

from data_generator import generate_dynamic_cases, generate_training_rows
from histogram_model import predict_histogram_cases, train_histogram_model
from model import (
    format_run_summary,
    load_model,
    make_run_summary,
    save_model,
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

# ── 路径常量 ──
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "histogram_model.joblib"


# ── 缓存 ──
@st.cache_resource
def get_cached_histogram_model(model_path: Path):
    """缓存直方图模型（避免重复加载）。"""
    return load_model(model_path)


@st.cache_data
def get_cached_predictions(predictions_path: Path):
    """缓存预测结果。"""
    if not predictions_path.exists():
        return None
    try:
        return json.loads(predictions_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ── 流水线 ──
def run_histogram_pipeline(
    group_config: dict, force_retrain: bool = False
) -> tuple[bool, str]:
    """运行直方图模型完整流水线（含进度条）。"""
    logs = []
    progress_bar = st.progress(0, text="初始化...")

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state["_current_run_signature"] = {
            "group_id": group_config.get("group_id"),
            "samples_per_label": group_config.get("samples_per_label"),
            "rf_params": dict(group_config.get("rf_params", {})),
            "engine": "histogram_uint8",
        }

        # Step 1: 检查缓存模型
        progress_bar.progress(10, text="检查缓存模型...")
        model = None
        if not force_retrain and MODEL_PATH.exists():
            model = get_cached_histogram_model(MODEL_PATH)
            if model:
                logs.append("✓ 已加载缓存直方图模型")

        # Step 2: 生成训练数据
        progress_bar.progress(25, text="生成训练数据...")
        training_rows = generate_training_rows(group_config)
        logs.append(f"✓ 生成 {len(training_rows)} 条训练样本")

        # Step 3: 训练模型（如果无缓存）
        if model is None:
            progress_bar.progress(50, text="训练直方图随机森林模型...")
            model, metrics = train_histogram_model(training_rows, group_config)
            logs.append(
                f"✓ 模型训练完成，准确率: {metrics['accuracy']:.4f}"
                f"（CV: {metrics.get('cv_mean_accuracy', 0):.4f}"
                f" ± {metrics.get('cv_std_accuracy', 0):.4f}）"
            )

            progress_bar.progress(65, text="保存模型...")
            save_model(model, MODEL_PATH)
            (OUTPUT_DIR / "model_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            st.cache_resource.clear()
            logs.append("✓ 模型已保存")
        else:
            metrics_path = OUTPUT_DIR / "model_metrics.json"
            metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {}
            )
            logs.append("✓ 已复用缓存直方图模型用于推理")

        # Step 4: 动态预测
        progress_bar.progress(80, text="动态生成预测用例...")
        input_cases = generate_dynamic_cases(group_config)
        if not input_cases:
            raise ValueError("未生成任何预测用例，请检查 case_blueprints")
        predicted_rows = predict_histogram_cases(model, input_cases)
        logs.append(f"✓ 完成 {len(predicted_rows)} 条预测")

        # Step 5: 保存结果
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

        # Step 6: 渲染图表
        progress_bar.progress(95, text="渲染图表...")
        render_status_chart(predicted_rows, OUTPUT_DIR)
        render_time_series_chart(predicted_rows, OUTPUT_DIR)
        render_stacked_bar_chart(predicted_rows, OUTPUT_DIR)
        render_feature_importance_chart(model, OUTPUT_DIR)
        logs.append("✓ 图表渲染完成")

        progress_bar.progress(100, text="完成!")
        return True, "\n".join(logs)

    except Exception as e:
        logger.error(f"Histogram pipeline failed: {e}", exc_info=True)
        progress_bar.empty()
        return False, f"错误: {str(e)}\n{chr(10).join(logs)}"


# ── 页面渲染 ──
def render_production_page(group_config: dict) -> None:
    """渲染生产看板页面（Page 1）。"""
    # ── 合规挂载点：引擎状态提示 ──
    st.info(
        "⚙️ **当前生产引擎**：直方图降维随机森林 (Accuracy: 0.97) "
        "| 🧪 **实验储备引擎**：基于 NLP 的时序语义分析器 (详见第二页效能评估)"
    )

    # ── 运行控制区 ──
    col1, col2 = st.columns([3, 1])
    with col1:
        run_clicked = st.button(
            "🚀 运行 / 刷新结果", type="primary", use_container_width=True
        )
    with col2:
        force_retrain = st.checkbox(
            "强制重训练",
            help="忽略缓存模型，使用当前参数重新训练（调整参数请用左侧'应用参数并重训练'按钮）",
        )

    if run_clicked:
        ok, logs = run_histogram_pipeline(group_config, force_retrain=force_retrain)
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
        with st.spinner("正在使用新参数重新训练直方图模型..."):
            ok, logs = run_histogram_pipeline(group_config, force_retrain=True)
            if ok:
                st.success("✅ 新参数重训练完成，结果已更新。")
                st.cache_data.clear()
            else:
                st.error(f"❌ 重训练失败：{logs}")
            if logs:
                with st.expander("📋 重训练日志", expanded=True):
                    st.text_area("logs", logs, height=200, label_visibility="collapsed")
            st.rerun()

    # ── 运行摘要 ──
    summary_txt = OUTPUT_DIR / "run_summary.txt"
    if summary_txt.exists():
        st.subheader("📊 运行摘要")
        st.code(summary_txt.read_text(encoding="utf-8"))
    else:
        st.info("👈 请先点击「运行 / 刷新结果」按钮。")

    # ── 预测结果表格 ──
    predictions = get_cached_predictions(OUTPUT_DIR / "predictions.json")
    if isinstance(predictions, list) and predictions:
        st.subheader("📋 预测结果")
        # 精选列并重命名为中文，过滤掉嵌套对象和原始特征值
        display_cols = [k for k in PREDICTION_COLUMNS_CN if k in predictions[0]]
        df = pd.DataFrame(predictions)[display_cols]
        df.columns = [PREDICTION_COLUMNS_CN[c] for c in display_cols]
        # 将时间戳格式化为更易读的形式
        if "时间戳" in df.columns:
            df["时间戳"] = pd.to_datetime(df["时间戳"]).dt.strftime("%Y-%m-%d %H:%M")
        # 用颜色标注预测标签（安全=绿，攻击=红）
        st.dataframe(
            df.style.map(
                lambda v: "color: #2e7d32; font-weight: 600" if v == "安全"
                else ("color: #c62828; font-weight: 600" if isinstance(v, str) and v else ""),
                subset=["预期标签", "预测标签"],
            ),
            use_container_width=True,
            height=400,
        )

    # ── 可视化图表 ──
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

    # ── 特征重要性 ──
    feature_importance_path = OUTPUT_DIR / "feature_importance.png"
    if feature_importance_path.exists():
        st.subheader("🔍 特征重要性")
        st.image(str(feature_importance_path), use_container_width=True)


# ── st.Page 入口：Streamlit 通过 nav.run() 执行此脚本时自动渲染 ──
_gc = st.session_state.get("group_config")
if _gc is None:
    st.error("无法加载配置，请检查 group_config.json")
else:
    render_production_page(_gc)
