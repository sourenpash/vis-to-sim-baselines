"""Temporal stability metrics for static-scene segments.

The locomotion-relevant question is: "how much does the depth wobble on a
scene that is not actually moving?". We answer it with two simple stats:

* ``temporal_std_mm`` - mean (over pixels valid in every frame) of the
  per-pixel standard deviation across the static window.
* ``frame_to_frame_diff_mm`` - mean of ``|d_t - d_{t-1}|`` over the same
  pixels, expressed in millimeters.

Both numbers are computed inside the locomotion-relevant near-field band
``[near_min_m, near_max_m]`` so faraway garbage doesn't dominate the mean.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _stack_valid_band(
    depths: List[np.ndarray],
    near_min_m: float,
    near_max_m: float,
) -> np.ndarray:
    """Returns a ``(T, H, W)`` float32 array; invalid/out-of-band pixels are NaN."""
    if not depths:
        return np.empty((0, 0, 0), dtype=np.float32)
    H, W = depths[0].shape
    out = np.empty((len(depths), H, W), dtype=np.float32)
    for t, d in enumerate(depths):
        d = d.astype(np.float32, copy=False)
        valid = np.isfinite(d) & (d >= near_min_m) & (d <= near_max_m)
        d_nan = np.where(valid, d, np.nan)
        out[t] = d_nan
    return out


def temporal_metrics(
    depths: List[np.ndarray],
    near_min_m: float = 0.2,
    near_max_m: float = 3.0,
    min_valid_fraction: float = 0.5,
) -> Dict[str, Any]:
    """Computes temporal stability metrics over a contiguous static segment.

    Parameters
    ----------
    depths
        Sequence of ``HxW`` float32 depth maps in meters. Should already be
        a static-scene segment.
    near_min_m, near_max_m
        Pixels outside this band are excluded.
    min_valid_fraction
        A pixel is included only if at least this fraction of frames had a
        valid depth there.
    """
    T = len(depths)
    if T < 2:
        return {
            "n_frames_used": int(T),
            "temporal_std_mm_mean": float("nan"),
            "frame_to_frame_diff_mm_mean": float("nan"),
        }

    stack = _stack_valid_band(depths, near_min_m, near_max_m)  # (T, H, W) NaN-padded
    valid_per_pixel = (~np.isnan(stack)).sum(axis=0)
    keep_mask = valid_per_pixel >= max(2, int(np.ceil(min_valid_fraction * T)))
    if not keep_mask.any():
        return {
            "n_frames_used": int(T),
            "temporal_std_mm_mean": float("nan"),
            "frame_to_frame_diff_mm_mean": float("nan"),
        }

    std_per_pixel = np.nanstd(stack, axis=0)  # meters
    diffs = np.abs(np.diff(stack, axis=0))    # (T-1, H, W) meters, NaN if either side NaN
    diff_per_pixel = np.nanmean(diffs, axis=0)

    std_kept = std_per_pixel[keep_mask]
    diff_kept = diff_per_pixel[keep_mask]

    return {
        "n_frames_used": int(T),
        "temporal_std_mm_mean": float(np.nanmean(std_kept) * 1000.0),
        "frame_to_frame_diff_mm_mean": float(np.nanmean(diff_kept) * 1000.0),
        "near_band_min_m": near_min_m,
        "near_band_max_m": near_max_m,
    }
