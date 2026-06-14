"""算法效能对比 PDF 报告生成器。

使用 fpdf2 生成包含实测数据、图表和分析结论的 PDF 报告。
中文字体: SimHei (Windows)。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fpdf import FPDF

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

_CHART_FILES = [
    ("perf_timing_compare.png", "训练 / 预测耗时对比"),
    ("perf_cache_hit_rate.png", "缓存命中率对比 (Intel VTune PMC 实测)"),
    ("perf_cpu_compare.png", "CPU 利用率对比"),
    ("perf_radar_compare.png", "综合效能雷达图"),
]


def _find_chinese_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# PDF 类
# ---------------------------------------------------------------------------
class _ReportPDF(FPDF):
    def __init__(self, font_path: str):
        super().__init__()
        self._font_path = font_path
        # 注册中文字体
        self.add_font("zh", "", font_path)
        self.add_font("zh", "B", font_path)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("zh", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "算法效能对比分析报告  |  CANary 金丝雀异常检测系统", align="R")
        self.ln(12)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("zh", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

    # --- 辅助方法 ---
    def section_title(self, title: str):
        self.set_font("zh", "B", 14)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, title, ln=1)
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def body_text(self, text: str):
        self.set_font("zh", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def kv_line(self, key: str, value: str, bold_val: bool = False):
        self.set_font("zh", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(70, 7, key, ln=0)
        style = "B" if bold_val else ""
        self.set_font("zh", style, 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, value, ln=1)

    def insert_chart(self, img_path: str | Path, caption: str):
        p = Path(img_path)
        if not p.exists():
            return
        # 检查是否需要新页
        if self.get_y() > 170:
            self.add_page()
        self.image(str(p), x=15, w=180)
        self.ln(2)
        self.set_font("zh", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"图: {caption}", align="C", ln=1)
        self.ln(4)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def generate_pdf_report(
    results: dict[str, Any],
    output_dir: Path,
    conclusion_text: str,
) -> bytes:
    """生成算法效能对比 PDF 报告，返回 PDF 二进制内容。

    Args:
        results: run_comparison() 返回的完整结果字典
        output_dir: 图表所在目录（含 perf_*.png）
        conclusion_text: PerfBenchmark.generate_conclusion() 生成的文本
    """
    font_path = _find_chinese_font()
    if not font_path:
        raise RuntimeError(
            "未找到中文字体文件，请安装 SimHei 或 Noto Sans CJK 字体"
        )

    pdf = _ReportPDF(font_path)
    pdf.add_page()

    # ── 封面标题 ──
    pdf.set_font("zh", "B", 22)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 15, "算法效能对比分析报告", align="C", ln=1)
    pdf.set_font("zh", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "传统 scikit-learn (float64) vs 直方图离散化 (uint8)", align="C",
             ln=1)
    pdf.ln(6)

    # ── 基本信息 ──
    cfg = results.get("config", {})
    hw_info = results.get("hardware_info", {})
    src_labels = results.get("source_labels", {})
    pdf.section_title("1. 测试配置")
    pdf.kv_line("样本总量", f"{cfg.get('n_samples', 0):,}")
    pdf.kv_line("特征维度", str(cfg.get("n_features", 0)))
    pdf.kv_line("每标签样本数", f"{cfg.get('samples_per_label', 0):,}")
    pdf.ln(2)

    # ── 硬件环境 ──
    if hw_info:
        pdf.set_font("zh", "B", 11)
        pdf.set_text_color(180, 60, 20)
        pdf.cell(0, 8, "硬件配置环境", ln=1)
        pdf.set_font("zh", "", 10)
        pdf.set_text_color(40, 40, 40)
        cpu_model = str(hw_info.get("cpu_model", "Unknown"))
        pdf.kv_line("CPU 型号", cpu_model)
        pdf.kv_line("CPU 物理核心", str(hw_info.get("cpu_cores_physical", "N/A")))
        pdf.kv_line("CPU 逻辑核心", str(hw_info.get("cpu_cores_logical", "N/A")))
        pdf.kv_line("L1D Cache", f"{hw_info.get('l1d_cache_kb', 0)} KB (每核)")
        pdf.kv_line("L2 Cache", f"{hw_info.get('l2_cache_kb', 0)} KB")
        pdf.kv_line("L3 Cache", f"{hw_info.get('l3_cache_kb', 0)} KB (共享)")
        pdf.kv_line("Cache Line", f"{hw_info.get('cache_line_bytes', 0)} B")
        pdf.kv_line("系统总内存", f"{hw_info.get('total_ram_gb', 0):.1f} GB")
        pdf.kv_line("Python 版本", str(hw_info.get("python_version", "N/A")))
        pdf.kv_line("NumPy 版本", str(hw_info.get("numpy_version", "N/A")))
        pdf.kv_line("scikit-learn 版本", str(hw_info.get("sklearn_version", "N/A")))
        pdf.ln(4)

    # ── 数据来源说明 ──
    if src_labels:
        pdf.set_font("zh", "B", 10)
        pdf.set_text_color(180, 60, 20)
        pdf.cell(0, 7, "数据来源说明", ln=1)
        pdf.set_font("zh", "", 10)
        pdf.set_text_color(40, 40, 40)
        src_map = [
            ("训练/预测耗时 & 准确率", src_labels.get("timing", "N/A")),
            ("CPU 利用率", src_labels.get("cpu", "N/A")),
            ("内存占用", src_labels.get("memory", "N/A")),
            ("数据访问速度", src_labels.get("access_speed", "N/A")),
            ("缓存命中率", src_labels.get("cache", "N/A")),
        ]
        for label, source in src_map:
            pdf.set_font("zh", "B", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(55, 7, label, ln=0)
            pdf.set_font("zh", "", 10)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 7, source, ln=1)
        pdf.ln(2)

    pdf.set_font("zh", "B", 10)
    pdf.set_text_color(180, 60, 20)
    pdf.cell(0, 7, "评测说明", ln=1)
    pdf.set_font("zh", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(
        0,
        6,
        "本报告中的性能指标主要来自程序实际运行后的统计结果，属于实测评测而非理论推导。"
        "实验结论应理解为当前数据规模、硬件环境和实现方式下的相对表现，并非绝对普适结论。",
    )
    pdf.ln(4)

    # ── 核心指标对比 ──
    sk = results.get("sklearn", {})
    hi = results.get("histogram", {})
    timing_src = src_labels.get("timing", "") if src_labels else ""
    cpu_src = src_labels.get("cpu", "") if src_labels else ""
    mem_src = src_labels.get("memory", "") if src_labels else ""
    pdf.section_title("2. 核心性能指标")
    if timing_src:
        pdf.set_font("zh", "", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, f"数据来源：{timing_src}（训练/预测耗时、准确率）, {cpu_src}（CPU利用率）",
                 ln=1)
        pdf.ln(2)

    pdf.kv_line("scikit-learn 训练耗时", f"{sk.get('train_time_sec', 0):.4f}s")
    pdf.kv_line("直方图算法训练耗时", f"{hi.get('train_time_sec', 0):.4f}s")
    sk_t = sk.get("train_time_sec", 0)
    hi_t = hi.get("train_time_sec", 0)
    if hi_t > 0 and sk_t > 0:
        ratio = sk_t / hi_t
        winner = "直方图快" if ratio >= 1 else "scikit-learn 快"
        pdf.kv_line("训练加速比", f"{winner} {max(ratio, 1/ratio):.2f}x", bold_val=True)

    pdf.kv_line("scikit-learn 预测耗时", f"{sk.get('predict_time_sec', 0):.4f}s")
    pdf.kv_line("直方图算法预测耗时", f"{hi.get('predict_time_sec', 0):.4f}s")

    pdf.kv_line("scikit-learn 准确率", f"{sk.get('accuracy', 0):.4f}")
    pdf.kv_line("直方图算法准确率", f"{hi.get('accuracy', 0):.4f}")
    pdf.ln(4)

    # ── 内存与缓存 ──
    mem = results.get("memory_analysis", {})
    acc_src = src_labels.get("access_speed", "") if src_labels else ""
    pdf.section_title("3. 内存占用与缓存行为")
    if mem_src or acc_src:
        pdf.set_font("zh", "", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, f"数据来源：{mem_src}（内存占用）, {acc_src}（数据访问速度）",
                 ln=1)
        pdf.ln(2)
    pdf.kv_line("float64 数据量", f"{mem.get('float64_total_kb', 0):.1f} KB")
    pdf.kv_line("uint8 数据量", f"{mem.get('uint8_total_kb', 0):.1f} KB")
    pdf.kv_line("内存缩减", f"{mem.get('memory_reduction_pct', 0):.1f}%", bold_val=True)

    # VTune 缓存数据
    sk_cache = sk.get("cache", {})
    hi_cache = hi.get("cache", {})
    cache_src = src_labels.get("cache", "") if src_labels else ""
    if sk_cache.get("measured"):
        pdf.ln(2)
        pdf.set_font("zh", "B", 11)
        pdf.set_text_color(180, 60, 20)
        pdf.cell(0, 8, f"Intel VTune PMC 硬件实测数据（数据来源：{cache_src}）",
                 ln=1)
        pdf.set_font("zh", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.kv_line("sklearn L1D 命中率", f"{sk_cache.get('l1d_hit_rate', 0)*100:.2f}%")
        pdf.kv_line("直方图 L1D 命中率", f"{hi_cache.get('l1d_hit_rate', 0)*100:.2f}%")
        pdf.kv_line("sklearn L3 命中率", f"{sk_cache.get('l3_hit_rate', 0)*100:.2f}%")
        pdf.kv_line("直方图 L3 命中率", f"{hi_cache.get('l3_hit_rate', 0)*100:.2f}%", bold_val=True)
        sk_l3m = sk_cache.get("l3_miss_count", 0)
        hi_l3m = hi_cache.get("l3_miss_count", 0)
        pdf.kv_line("sklearn L3 Miss 计数", f"{sk_l3m:,}")
        pdf.kv_line("直方图 L3 Miss 计数", f"{hi_l3m:,}")
        if hi_l3m > 0:
            pdf.kv_line("L3 Miss 比值",
                        f"sklearn 是直方图的 {sk_l3m/hi_l3m:.1f}x",
                        bold_val=True)
    elif sk_cache:
        pdf.kv_line(f"sklearn L1D 命中率（{cache_src}）",
                     f"{sk_cache.get('l1d_hit_rate', 0)*100:.2f}%")
        pdf.kv_line(f"直方图 L1D 命中率（{cache_src}）",
                     f"{hi_cache.get('l1d_hit_rate', 0)*100:.2f}%")
    pdf.ln(4)

    # ── 图表 ──
    pdf.section_title("4. 可视化图表")
    for fname, caption in _CHART_FILES:
        fpath = output_dir / fname
        if fpath.exists():
            pdf.insert_chart(fpath, caption)

    # ── 分析结论 ──
    pdf.add_page()
    pdf.section_title("5. 详细分析结论")
    pdf.set_font("zh", "B", 10)
    pdf.set_text_color(180, 60, 20)
    pdf.cell(0, 7, "结论边界说明", ln=1)
    pdf.set_font("zh", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(
        0,
        6,
        "以下结论基于当前实验条件下的实测数据，已结合控制变量分析对算法、数据类型和内存布局影响进行区分。"
        "由于硬件平台、库版本、线程设置以及数据分布均会影响结果，因此结论应作为参考性的性能评估。",
    )
    pdf.ln(2)

    # 替换可能在字体中缺失的 Unicode 字符
    _BOX_MAP = str.maketrans({
        "│": "|", "┼": "+", "─": "-",
        "═": "=", "┐": "+", "┌": "+",
        "└": "+", "┘": "+", "┬": "+",
        "┴": "+", "├": "+", "┤": "+",
    })

    for line in conclusion_text.split("\n"):
        stripped = line.strip().translate(_BOX_MAP)
        if not stripped:
            pdf.ln(3)
            continue
        if stripped.startswith("="):
            continue  # 跳过装饰线
        # 检测小标题
        if stripped.startswith("[【") or stripped.startswith("【"):
            if stripped.endswith("】"):
                pdf.set_font("zh", "B", 11)
                pdf.set_text_color(30, 60, 120)
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, 8, stripped, ln=1)
                pdf.set_font("zh", "", 10)
                pdf.set_text_color(40, 40, 40)
                continue
        # 普通文本行
        pdf.set_x(pdf.l_margin)
        try:
            pdf.multi_cell(0, 5.5, stripped)
        except Exception:
            # 如果仍然失败，截短后重试
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, stripped[:80])

    # ── 输出 ──
    raw = pdf.output()
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return raw.encode("latin-1")
