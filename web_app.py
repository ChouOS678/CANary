"""CANary — 基于 NLP 与微架构优化的 CAN 总线时序语义异常检测 - Streamlit 主控入口。

架构：单侧边栏全局控制 + 双页面按需调用（惰性加载）。

- Page 1 (生产看板): 仅调用 histogram_model.py，绝对禁止 NLP 污染
- Page 2 (Benchmark Lab): 调用 perf_analysis.py 进行 A/B/C 三路打擂

侧边栏参数作为全局 Session State 存在，无论在 Page 1 还是 Page 2，
修改参数后都应触发对应页面的重新计算，且不能互相干扰。
"""

from __future__ import annotations

import glob as _glob
import json
from pathlib import Path

import streamlit as st

from config import LABELS

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "histogram_model.joblib"
PERF_DIR = OUTPUT_DIR / "perf_compare"


# ========== 全局配置 ==========

st.set_page_config(page_title="CANary - 金丝雀 | CAN 总线时序语义异常检测", layout="wide", page_icon="🛡️")
st.title("🛡️ CANary — 金丝雀时序语义异常检测可视化系统")


# ========== 侧边栏 (全局控制) ==========

def _load_group_config() -> dict | None:
    """从 group_config.json 加载配置。"""
    config_path = BASE_DIR / "group_config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


with st.sidebar:
    st.header("⚙️ 参数控制")
    group_config = _load_group_config()

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
            50,
            10000,
            int(group_config.get("samples_per_label", 320)),
            step=50,
            help="总样本量 = 每标签样本数 × 标签数（当前 6 个攻击类别）",
        )
        total_samples = samples_per_label * n_labels
        if total_samples <= 5000:
            size_hint = "🟢 小数据量"
        elif total_samples <= 30000:
            size_hint = "🟡 中数据量"
        else:
            size_hint = "🔴 大数据量"
        st.caption(f"预估总样本量：{total_samples:,}（{size_hint}）")

        if st.button("应用参数并重训练", type="secondary"):
            # 更新配置对象
            group_config["rf_params"]["n_estimators"] = n_estimators
            group_config["rf_params"]["max_depth"] = max_depth
            group_config["samples_per_label"] = samples_per_label
            # 持久化到文件
            (BASE_DIR / "group_config.json").write_text(
                json.dumps(group_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # 删除缓存模型文件，强制用新参数重训练
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
            # 清除旧性能对比数据（参数变了，旧对比结果已失效）
            for _stale_file in [
                PERF_DIR / "comparison_results.json",
                PERF_DIR / "perf_analysis.txt",
                PERF_DIR / "算法效能对比报告.pdf",
            ]:
                if _stale_file.exists():
                    _stale_file.unlink()
            for _chart in _glob.glob(str(PERF_DIR / "perf_*.png")):
                Path(_chart).unlink()
            # 清除旧 PDF 缓存
            st.session_state.pop("_pdf_report_bytes", None)
            # 清除 Streamlit 缓存
            st.cache_resource.clear()
            st.cache_data.clear()
            # 设置 session_state 标记，触发 rerun 后自动执行 pipeline
            st.session_state["_auto_retrain"] = True
            st.rerun()

        # ── 将参数写入 Session State（全局共享）──
        st.session_state["group_config"] = group_config
        st.session_state["n_estimators"] = n_estimators
        st.session_state["max_depth"] = max_depth
        st.session_state["samples_per_label"] = samples_per_label


# ========== 主内容区 ==========

if group_config is None:
    st.error("无法加载配置文件，请检查 group_config.json")
    st.stop()

# ── 双页面路由（惰性加载：仅渲染当前页面，不预计算另一页面）──
page_production = st.Page(
    "page_production.py",
    title="训练与预测结果",
    icon="📊",
)
page_benchmark = st.Page(
    "page_benchmark.py",
    title="算法性能评估",
    icon="⚡",
)

nav = st.navigation(
    {
        "生产看板": [page_production],
        "实验室": [page_benchmark],
    },
    position="top",
)
nav.run()
