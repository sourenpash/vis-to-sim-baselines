"""Runtime metrics derived from ``timings.csv``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def runtime_metrics(timings_csv: Path) -> Dict[str, Any]:
    """Latency percentiles, FPS, jitter, and peak GPU memory."""
    timings_csv = Path(timings_csv)
    if not timings_csv.exists():
        return {"n_frames": 0}

    df = pd.read_csv(timings_csv)
    if len(df) == 0:
        return {"n_frames": 0}

    infer = df["infer_ms"].to_numpy(dtype=np.float64)
    total = df["total_ms"].to_numpy(dtype=np.float64)
    gpu = df["gpu_mem_mb"].to_numpy(dtype=np.float64) if "gpu_mem_mb" in df else np.zeros_like(total)

    # Drop the first frame from latency stats (warm-up tail) only when we have
    # enough samples for the trim to be meaningful.
    trim = 1 if len(total) >= 5 else 0

    out: Dict[str, Any] = {
        "n_frames": int(len(df)),
        "infer_ms_p50": float(np.percentile(infer[trim:], 50)) if len(infer) > trim else float("nan"),
        "infer_ms_p90": float(np.percentile(infer[trim:], 90)) if len(infer) > trim else float("nan"),
        "infer_ms_mean": float(np.mean(infer[trim:])) if len(infer) > trim else float("nan"),
        "total_ms_p50": float(np.percentile(total[trim:], 50)) if len(total) > trim else float("nan"),
        "total_ms_mean": float(np.mean(total[trim:])) if len(total) > trim else float("nan"),
        "fps_from_total": float(1000.0 / max(np.mean(total[trim:]), 1e-6)) if len(total) > trim else float("nan"),
        "jitter_ms_std": float(np.std(total[trim:])) if len(total) > trim else float("nan"),
        "gpu_mem_mb_peak": float(np.max(gpu)) if len(gpu) else 0.0,
    }
    return out
