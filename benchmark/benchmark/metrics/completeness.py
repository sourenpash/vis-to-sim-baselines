"""Completeness metrics: fraction of pixels with usable depth."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


def _is_finite_positive(d: np.ndarray) -> np.ndarray:
    return np.isfinite(d) & (d > 0)


def completeness_metrics(
    depths: Iterable[np.ndarray],
    near_min_m: float = 0.2,
    near_max_m: float = 3.0,
    roi: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Compute mean valid-pixel ratios across a sequence of depth maps.

    ``roi`` is an optional dict ``{x, y, w, h}`` (top-left + size, in pixels)
    restricting the count to a region.
    """
    valid_total = 0
    in_band_total = 0
    pixels_total = 0

    for d in depths:
        if d is None:
            continue
        if roi is not None:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            d = d[y : y + h, x : x + w]
        valid = _is_finite_positive(d)
        in_band = valid & (d >= near_min_m) & (d <= near_max_m)
        valid_total += int(valid.sum())
        in_band_total += int(in_band.sum())
        pixels_total += int(d.size)

    if pixels_total == 0:
        return {
            "valid_pct": float("nan"),
            "near_band_valid_pct": float("nan"),
            "near_band_min_m": near_min_m,
            "near_band_max_m": near_max_m,
        }

    return {
        "valid_pct": 100.0 * valid_total / pixels_total,
        "near_band_valid_pct": 100.0 * in_band_total / pixels_total,
        "near_band_min_m": near_min_m,
        "near_band_max_m": near_max_m,
    }
