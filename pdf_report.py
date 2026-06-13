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
        self.add_font("zh", "", font_path, uni=True)
        self.add_font("zh", "B", font_path, uni=True)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("zh", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "算法效能对比分析报告  |  CAN 总线异常检测系统", align="R")
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
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
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
        self.cell(70, 7, key, new_x="END")
        style = "B" if bold_val else ""
        self.set_font("zh", style, 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

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
        self.cell(0, 6, f"图: {caption}", align="C", new_x="LMARGIN", new_y="NEXT")
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
    pdf.cell(0, 15, "算法效能对比分析报告", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("zh", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "传统 scikit-learn (float64) vs 直方图离散化 (uint8)", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── 基本信息 ──
    cfg = results.get("config", {})
    pdf.section_title("1. 测试配置")
    pdf.kv_line("样本总量", f"{cfg.get('n_samples', 0):,}")
    pdf.kv_line("特征维度", str(cfg.get("n_features", 0)))
    pdf.kv_line("每标签样本数", f"{cfg.get('samples_per_label', 0):,}")
    pdf.ln(4)

    # ── 核心指标对比 ──
    sk = results.get("sklearn", {})
    hi = results.get("histogram", {})
    pdf.section_title("2. 核心性能指标")

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
    pdf.section_title("3. 内存占用与缓存行为")
    pdf.kv_line("float64 数据量", f"{mem.get('float64_total_kb', 0):.1f} KB")
    pdf.kv_line("uint8 数据量", f"{mem.get('uint8_total_kb', 0):.1f} KB")
    pdf.kv_line("内存缩减", f"{mem.get('memory_reduction_pct', 0):.1f}%", bold_val=True)

    # VTune 缓存数据
    sk_cache = sk.get("cache", {})
    hi_cache = hi.get("cache", {})
    if sk_cache.get("measured"):
        pdf.ln(2)
        pdf.set_font("zh", "B", 11)
        pdf.set_text_color(180, 60, 20)
        pdf.cell(0, 8, "Intel VTune PMC 硬件实测数据", new_x="LMARGIN", new_y="NEXT")
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
        pdf.kv_line("sklearn L1D 命中率 (理论)",
                     f"{sk_cache.get('l1d_hit_rate', 0)*100:.2f}%")
        pdf.kv_line("直方图 L1D 命中率 (理论)",
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
                pdf.cell(0, 8, stripped, new_x="LMARGIN", new_y="NEXT")
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
    return bytes(pdf.output())
