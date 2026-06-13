"""Visualization functions for charts (Matplotlib + optional Plotly)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import FEATURES, FEATURES_CN, LABELS


def configure_matplotlib() -> None:
    """Configure matplotlib for Chinese fonts."""
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def render_status_chart(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Render donut chart of normal vs abnormal packets."""
    configure_matplotlib()

    normal_packets = sum(int(row["normal_packets"]) for row in rows)
    abnormal_packets = sum(int(row["abnormal_packets"]) for row in rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor="#0f1b2f")
    ax.pie(
        [normal_packets, abnormal_packets],
        labels=["正常流量", "异常攻击"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#06c98e", "#ff3b4d"],
        wedgeprops={"width": 0.42, "edgecolor": "#0f1b2f", "linewidth": 4},
        textprops={"color": "#e9f4ff", "fontsize": 12},
    )
    ax.set_title("攻击/异常状态统计图", fontsize=18, color="#e9f4ff", pad=14)
    fig.savefig(
        output_dir / "status_donut.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def render_time_series_chart(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Render timeline chart of attack intensity."""
    configure_matplotlib()

    labels = [str(item["timestamp"])[11:16] for item in rows]
    values = [float(item["attack_intensity"]) for item in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.0), facecolor="#0f1b2f")
    ax.set_facecolor("#15233a")
    ax.plot(
        labels,
        values,
        color="#1e88ff",
        linewidth=3.4,
        marker="o",
        markersize=7,
        markerfacecolor="#ffffff",
        markeredgewidth=2.6,
    )
    ax.fill_between(labels, values, color="#1e88ff", alpha=0.18)
    ax.set_ylim(0, 50)
    ax.grid(axis="y", color="white", alpha=0.08, linewidth=1)
    ax.tick_params(colors="#c2d4ea")
    for spine in ax.spines.values():
        spine.set_color("#274160")
    ax.set_title("攻击时间分布图", fontsize=18, color="#e9f4ff", pad=12)
    fig.savefig(
        output_dir / "attack_timeline.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def render_stacked_bar_chart(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Render stacked bar chart of attack severity by type."""
    configure_matplotlib()

    ordered_types = [label for label in LABELS if label != "安全"]
    low_values: list[int] = []
    medium_values: list[int] = []
    high_values: list[int] = []

    for attack_type in ordered_types:
        selected = [row for row in rows if str(row["predicted_label"]) == attack_type]
        low_values.append(sum(int(row["severity_split"]["low"]) for row in selected))
        medium_values.append(
            sum(int(row["severity_split"]["medium"]) for row in selected)
        )
        high_values.append(sum(int(row["severity_split"]["high"]) for row in selected))

    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="#0f1b2f")
    ax.set_facecolor("#15233a")
    x_axis = np.arange(len(ordered_types))
    width = 0.42
    ax.bar(x_axis, low_values, width=width, color="#ff4d6d", label="低等级")
    ax.bar(
        x_axis, medium_values, width=width, bottom=low_values, color="#ff7a1a", label="中等级"
    )
    ax.bar(
        x_axis,
        high_values,
        width=width,
        bottom=np.array(low_values) + np.array(medium_values),
        color="#ffbf4d",
        label="高等级",
    )
    ax.set_xticks(x_axis, ordered_types)
    ax.grid(axis="y", color="white", alpha=0.08, linewidth=1)
    ax.tick_params(colors="#c2d4ea")
    for spine in ax.spines.values():
        spine.set_color("#274160")
    ax.legend(frameon=False, labelcolor="#e9f4ff")
    ax.set_title("攻击等级/类型分布图", fontsize=18, color="#e9f4ff", pad=12)
    fig.savefig(
        output_dir / "attack_type_stacked.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def render_feature_importance_chart(
    model, output_dir: Path, top_n: int | None = None
) -> None:
    """Render feature importance chart with Chinese labels and value annotations."""
    configure_matplotlib()

    importances = model.feature_importances_
    n = top_n if top_n is not None else len(FEATURES)
    indices = np.argsort(importances)[::-1][:n]
    cn_labels = [FEATURES_CN.get(FEATURES[i], FEATURES[i]) for i in indices]
    bar_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1b2f")
    ax.set_facecolor("#15233a")
    bars = ax.bar(
        range(len(indices)),
        bar_values,
        color="#1e88ff",
        edgecolor="#0f1b2f",
    )
    # 在柱状图顶部标注具体数值
    for bar, val in zip(bars, bar_values):
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#e9f4ff",
            fontsize=8,
        )
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(cn_labels, rotation=45, ha="right")
    ax.tick_params(colors="#c2d4ea")
    for spine in ax.spines.values():
        spine.set_color("#274160")
    ax.set_title("特征重要性排名", fontsize=18, color="#e9f4ff", pad=12)
    ax.set_ylabel("重要性", color="#e9f4ff")
    fig.tight_layout()
    fig.savefig(
        output_dir / "feature_importance.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


# Optional: Plotly interactive charts (if plotly is installed)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True

    def render_interactive_timeline(rows: list[dict[str, object]]) -> go.Figure:
        """Create interactive timeline chart with Plotly."""
        timestamps = [str(row["timestamp"])[11:16] for row in rows]
        intensities = [float(row["attack_intensity"]) for row in rows]
        labels = [str(row["predicted_label"]) for row in rows]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=intensities,
                mode="lines+markers",
                name="攻击强度",
                line=dict(color="#1e88ff", width=3),
                marker=dict(size=8),
                text=labels,
                hovertemplate="<b>%{x}</b><br>强度: %{y:.1f}<br>类型: %{text}<extra></extra>",
            )
        )
        fig.update_layout(
            title="攻击时间分布图（交互式）",
            xaxis_title="时间",
            yaxis_title="攻击强度",
            template="plotly_dark",
            height=400,
        )
        return fig

    def render_interactive_pie(rows: list[dict[str, object]]) -> go.Figure:
        """Create interactive pie chart with Plotly."""
        normal = sum(int(row["normal_packets"]) for row in rows)
        abnormal = sum(int(row["abnormal_packets"]) for row in rows)

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["正常流量", "异常攻击"],
                    values=[normal, abnormal],
                    hole=0.4,
                    marker=dict(colors=["#06c98e", "#ff3b4d"]),
                )
            ]
        )
        fig.update_layout(
            title="流量状态统计（交互式）",
            template="plotly_dark",
            height=400,
        )
        return fig

except ImportError:
    PLOTLY_AVAILABLE = False
