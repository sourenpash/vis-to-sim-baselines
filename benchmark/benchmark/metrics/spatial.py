"""Spatial noise metrics that quantify how flat a flat surface looks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


def _depth_to_xyz(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    x = (j - cx) * depth_m / fx
    y = (i - cy) * depth_m / fy
    z = depth_m
    return np.stack([x, y, z], axis=-1)


def _ransac_plane(
    points: np.ndarray,
    n_iters: int = 100,
    inlier_threshold_m: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[str, Any]]:
    """Fit a plane ``ax + by + cz + d = 0`` (unit normal) via RANSAC."""
    if rng is None:
        rng = np.random.default_rng(0)
    if points.shape[0] < 3:
        return None

    best_inliers = -1
    best_plane = None
    best_residuals = None
    for _ in range(n_iters):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(np.dot(n, p0))
        residuals = np.abs(points @ n + d)
        inliers = int((residuals < inlier_threshold_m).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (n, d)
            best_residuals = residuals

    if best_plane is None:
        return None
    n, d = best_plane
    return {
        "normal": n,
        "offset": float(d),
        "n_inliers": int(best_inliers),
        "n_points": int(points.shape[0]),
        "residuals": best_residuals,
    }


def plane_fit_metrics(
    depth_m: np.ndarray,
    K: np.ndarray,
    roi: Optional[Dict[str, int]],
    inlier_threshold_m: float = 0.01,
) -> Dict[str, Any]:
    """Fit a plane to ``roi`` of one depth map; returns RMSE in millimeters.

    ``roi`` is ``{x, y, w, h}``. If ``None``, uses the whole frame (rarely a
    good idea for a clean planar surface; supply an ROI when possible).
    """
    if roi is None:
        H, W = depth_m.shape
        roi = {"x": 0, "y": 0, "w": W, "h": H}

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    sub_depth = depth_m[y : y + h, x : x + w]
    K_sub = K.copy()
    K_sub[0, 2] -= x
    K_sub[1, 2] -= y

    valid = np.isfinite(sub_depth) & (sub_depth > 0)
    if valid.sum() < 200:
        return {
            "plane_rmse_mm": float("nan"),
            "plane_inlier_pct": float("nan"),
            "n_points": int(valid.sum()),
            "normal": [float("nan")] * 3,
        }

    xyz = _depth_to_xyz(sub_depth, K_sub)[valid]
    fit = _ransac_plane(xyz, inlier_threshold_m=inlier_threshold_m)
    if fit is None:
        return {
            "plane_rmse_mm": float("nan"),
            "plane_inlier_pct": float("nan"),
            "n_points": int(xyz.shape[0]),
            "normal": [float("nan")] * 3,
        }

    inlier_mask = fit["residuals"] < inlier_threshold_m
    inlier_residuals = fit["residuals"][inlier_mask]
    rmse_m = float(np.sqrt(np.mean(inlier_residuals ** 2))) if inlier_residuals.size else float("nan")
    return {
        "plane_rmse_mm": rmse_m * 1000.0,
        "plane_inlier_pct": 100.0 * fit["n_inliers"] / max(fit["n_points"], 1),
        "n_points": fit["n_points"],
        "normal": fit["normal"].tolist(),
    }


def aggregate_plane_fit(
    depths: Iterable[np.ndarray],
    K: np.ndarray,
    roi: Optional[Dict[str, int]],
    inlier_threshold_m: float = 0.01,
) -> Dict[str, Any]:
    """Mean plane-fit RMSE + std of plane normal across a sequence of frames."""
    rmses: list = []
    inlier_pcts: list = []
    normals: list = []
    for d in depths:
        m = plane_fit_metrics(d, K, roi, inlier_threshold_m=inlier_threshold_m)
        rmses.append(m["plane_rmse_mm"])
        inlier_pcts.append(m["plane_inlier_pct"])
        if not any(np.isnan(c) for c in m["normal"]):
            normals.append(m["normal"])

    rmses = np.asarray(rmses, dtype=np.float64)
    inlier_pcts = np.asarray(inlier_pcts, dtype=np.float64)

    if len(normals) >= 2:
        normals = np.asarray(normals)
        # Align signs to a reference normal so std reflects actual rotation,
        # not orientation flips.
        ref = normals[0]
        signs = np.sign((normals * ref).sum(axis=1))
        signs[signs == 0] = 1
        normals = normals * signs[:, None]
        mean_n = normals.mean(axis=0)
        mean_n /= max(np.linalg.norm(mean_n), 1e-9)
        cosang = np.clip((normals * mean_n).sum(axis=1), -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cosang))
        normal_std_deg = float(np.std(ang_deg))
    else:
        normal_std_deg = float("nan")

    return {
        "plane_rmse_mm_mean": float(np.nanmean(rmses)) if len(rmses) else float("nan"),
        "plane_rmse_mm_p90": float(np.nanpercentile(rmses, 90)) if len(rmses) else float("nan"),
        "plane_inlier_pct_mean": float(np.nanmean(inlier_pcts)) if len(inlier_pcts) else float("nan"),
        "plane_normal_std_deg": normal_std_deg,
        "n_frames": int(len(rmses)),
    }
