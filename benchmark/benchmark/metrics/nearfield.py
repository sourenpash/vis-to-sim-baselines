"""Locomotion-relevant near-field metrics.

These reuse :mod:`benchmark.metrics.spatial` for plane fitting and
:mod:`benchmark.metrics.completeness` for the near-band valid ratio. The
"ground plane" is just the bottom-third of the frame by default; users
can override with an explicit ROI.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from .spatial import aggregate_plane_fit


def _default_ground_roi(width: int, height: int) -> Dict[str, int]:
    """Bottom third of the image, full width."""
    return {
        "x": 0,
        "y": int(2 * height / 3),
        "w": int(width),
        "h": int(height - 2 * height / 3),
    }


def ground_plane_metrics(
    depths: Iterable[np.ndarray],
    K: np.ndarray,
    width: int,
    height: int,
    roi: Optional[Dict[str, int]] = None,
    inlier_threshold_m: float = 0.02,
) -> Dict[str, Any]:
    """Aggregate ground-plane fit stats over a sequence of frames.

    Returns the mean RMSE of the inliers (a stability proxy) and the std of
    the plane normal in degrees over time (locomotion stability proxy).
    """
    if roi is None:
        roi = _default_ground_roi(width, height)
    out = aggregate_plane_fit(depths, K, roi, inlier_threshold_m=inlier_threshold_m)
    out["roi"] = roi
    return {
        "ground_plane_rmse_mm_mean": out["plane_rmse_mm_mean"],
        "ground_plane_normal_std_deg": out["plane_normal_std_deg"],
        "ground_plane_inlier_pct_mean": out["plane_inlier_pct_mean"],
        "ground_roi": roi,
    }
