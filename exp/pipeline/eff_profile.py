# research/EPFlab/exp/pipeline/eff_profile.py
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


def _mb(x_bytes: int) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)


def _cpu_rss_mb() -> float:
    if psutil is None:
        return float("nan")
    try:
        return _mb(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return float("nan")


def _gpu_available() -> bool:
    return (torch is not None) and bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()


def _gpu_reset_peak(do_empty_cache: bool = False):
    if not _gpu_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
        if do_empty_cache:
            torch.cuda.empty_cache()
    except Exception:
        pass


def _gpu_sync():
    if not _gpu_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def _gpu_peak_mb() -> Tuple[float, float]:
    """return (peak_alloc_mb, peak_reserved_mb)"""
    if not _gpu_available():
        return (float("nan"), float("nan"))
    try:
        alloc = _mb(int(torch.cuda.max_memory_allocated()))
        reserv = _mb(int(torch.cuda.max_memory_reserved()))
        return (alloc, reserv)
    except Exception:
        return (float("nan"), float("nan"))


def _append_eff_csv(csv_path: Path, row: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


@contextmanager
def profile_stage(
    *,
    stage: str,
    args,
    eff_csv: Optional[Path] = None,
    do_empty_cache: bool = False,
):
    """
    Print per-stage:
      - wall time
      - CPU RSS (MB)
      - GPU peak allocated/reserved (MB)
    Optionally append to a CSV.

    Note:
      - For fair "realistic" timing, set do_empty_cache=False.
      - If you want clean peak mem snapshots, set do_empty_cache=True.
    """
    tag = str(getattr(args, "tag", "pipeline"))
    save_dir = str(getattr(args, "save_dir", "."))
    gpu = _gpu_available()

    cpu0 = _cpu_rss_mb()
    _gpu_sync()
    if gpu:
        _gpu_reset_peak(do_empty_cache=do_empty_cache)
    t0 = time.perf_counter()

    yield

    _gpu_sync()
    t1 = time.perf_counter()
    cpu1 = _cpu_rss_mb()
    peak_alloc, peak_reserv = _gpu_peak_mb()

    wall_s = t1 - t0
    cpu_delta = cpu1 - cpu0 if (np.isfinite(cpu0) and np.isfinite(cpu1)) else float("nan")

    print(
        f"[eff] stage={stage} | wall={wall_s:.3f}s | "
        f"cpu_rss={cpu1:.1f}MB (Δ{cpu_delta:.1f}MB) | "
        f"gpu_peak_alloc={peak_alloc:.1f}MB | gpu_peak_reserv={peak_reserv:.1f}MB | "
        f"tag={tag}"
    )

    if eff_csv is not None:
        row = {
            "tag": tag,
            "save_dir": save_dir,
            "stage": stage,
            "wall_s": float(wall_s),
            "cpu_rss_mb": float(cpu1),
            "cpu_delta_mb": float(cpu_delta),
            "gpu_peak_alloc_mb": float(peak_alloc),
            "gpu_peak_reserved_mb": float(peak_reserv),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_eff_csv(eff_csv, row)
