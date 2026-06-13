"""Utility functions including logging and helpers."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import psutil


def setup_logger(name: str = "anomaly_detector", level: int = logging.INFO) -> logging.Logger:
    """Create and configure logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def clamp_feature(feature: str, value: float, feature_bounds: dict) -> float:
    """Clamp feature value to valid range."""
    import numpy as np
    low, high = feature_bounds[feature]
    return float(np.clip(value, low, high))


def to_counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    """Count occurrences of a key in rows."""
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


class HardwareMonitor:
    """Background hardware utilization monitor.

    Samples CPU (per-core), system memory, and process memory at a fixed
    interval in a daemon thread. Call ``start()`` before training and
    ``stop()`` after training to collect a summary.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())

        # Sample storage
        self._cpu_total: list[float] = []
        self._cpu_per_core: list[list[float]] = []
        self._mem_percent: list[float] = []
        self._proc_mem_mb: list[float] = []
        self._timestamps: list[float] = []

    # ------------------------------------------------------------------
    def _sample_loop(self) -> None:
        num_cores = psutil.cpu_count(logical=True) or 1
        while not self._stop_event.is_set():
            self._timestamps.append(time.time())
            # CPU
            self._cpu_total.append(psutil.cpu_percent(interval=None))
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            if not self._cpu_per_core:
                self._cpu_per_core = [[] for _ in range(num_cores)]
            for i, val in enumerate(per_core):
                if i < num_cores:
                    self._cpu_per_core[i].append(val)
            # System memory
            self._mem_percent.append(psutil.virtual_memory().percent)
            # Process memory (RSS in MB)
            try:
                rss_mb = self._process.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss_mb = 0.0
            self._proc_mem_mb.append(rss_mb)

            self._stop_event.wait(self._interval)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start background monitoring."""
        psutil.cpu_percent(interval=None)  # prime the pump
        psutil.cpu_percent(interval=None, percpu=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("[HardwareMonitor] started (interval=%.1fs)", self._interval)

    def stop(self) -> dict[str, object]:
        """Stop monitoring and return a summary dict."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

        elapsed = (
            self._timestamps[-1] - self._timestamps[0]
            if len(self._timestamps) >= 2
            else 0.0
        )

        def _stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"avg": 0.0, "peak": 0.0, "min": 0.0}
            return {
                "avg": round(float(sum(values)) / len(values), 1),
                "peak": round(float(max(values)), 1),
                "min": round(float(min(values)), 1),
            }

        core_stats = [
            _stats(core) for core in self._cpu_per_core
        ] if self._cpu_per_core else []

        summary: dict[str, object] = {
            "duration_sec": round(elapsed, 2),
            "samples": len(self._timestamps),
            "cpu_total_pct": _stats(self._cpu_total),
            "cpu_per_core_pct": core_stats,
            "mem_system_pct": _stats(self._mem_percent),
            "proc_memory_mb": _stats(self._proc_mem_mb),
        }
        logger.info("[HardwareMonitor] stopped")
        return summary

    # ------------------------------------------------------------------
    @staticmethod
    def format_summary(summary: dict[str, object]) -> str:
        """Format hardware summary as readable text."""
        lines: list[str] = ["硬件利用率摘要："]
        lines.append(f"  监控时长: {summary['duration_sec']}s（{summary['samples']} 个采样点）")

        cpu = summary["cpu_total_pct"]
        if isinstance(cpu, dict):
            lines.append(f"  CPU 总利用率: 均值 {cpu['avg']}% / 峰值 {cpu['peak']}% / 最低 {cpu['min']}%")

        per_core = summary.get("cpu_per_core_pct", [])
        if isinstance(per_core, list):
            for i, c in enumerate(per_core):
                if isinstance(c, dict):
                    lines.append(f"    核心 {i}: 均值 {c['avg']}% / 峰值 {c['peak']}%")

        mem = summary["mem_system_pct"]
        if isinstance(mem, dict):
            lines.append(f"  系统内存: 均值 {mem['avg']}% / 峰值 {mem['peak']}%")

        pmem = summary["proc_memory_mb"]
        if isinstance(pmem, dict):
            lines.append(f"  进程内存: 均值 {pmem['avg']}MB / 峰值 {pmem['peak']}MB")

        return "\n".join(lines)
