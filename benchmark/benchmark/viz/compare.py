"""Side-by-side comparison images across adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np


def colorize_depth(
    depth_m: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    d = depth_m.copy()
    d[~valid] = 0.0
    if vmin is None:
        vmin = float(d[valid].min()) if valid.any() else 0.0
    if vmax is None:
        vmax = float(d[valid].max()) if valid.any() else 1.0
    norm = np.clip((d - vmin) / max(vmax - vmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, colormap)
    out[~valid] = (0, 0, 0)
    return out


def label_image(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _load_depth(npy_path: Path) -> Optional[np.ndarray]:
    if not npy_path.exists():
        return None
    return np.load(npy_path)


def _resize_to(img: np.ndarray, target_hw: tuple) -> np.ndarray:
    if img.shape[:2] == target_hw:
        return img
    return cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


def make_compare_image(
    rgb_bgr: np.ndarray,
    raw_depth_m: np.ndarray,
    per_adapter_depth: Dict[str, Optional[np.ndarray]],
    vmin: float = 0.2,
    vmax: float = 5.0,
) -> np.ndarray:
    """Return a tiled BGR image: RGB + raw_depth + each adapter's depth."""
    H, W = rgb_bgr.shape[:2]
    tiles: List[np.ndarray] = []
    tiles.append(label_image(rgb_bgr, "rgb"))

    raw_color = colorize_depth(raw_depth_m, vmin=vmin, vmax=vmax)
    raw_color = _resize_to(raw_color, (H, W))
    tiles.append(label_image(raw_color, "raw_depth (D435i HW)"))

    for name, d in per_adapter_depth.items():
        if d is None:
            blank = np.zeros((H, W, 3), dtype=np.uint8)
            tiles.append(label_image(blank, f"{name} (missing)"))
            continue
        col = colorize_depth(d, vmin=vmin, vmax=vmax)
        col = _resize_to(col, (H, W))
        tiles.append(label_image(col, name))

    return np.concatenate(tiles, axis=1)


def write_compare_grid(
    session_root: Path,
    frame_idx: int,
    adapters: Iterable[str],
    out_dir: Path,
    vmin: float = 0.2,
    vmax: float = 5.0,
) -> Optional[Path]:
    """Build a comparison image for one frame and save it under ``out_dir``."""
    rgb_path = session_root / "color" / f"{frame_idx:06d}.png"
    depth_png = session_root / "depth" / f"{frame_idx:06d}.png"
    if not rgb_path.exists() or not depth_png.exists():
        return None

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    raw_d_u16 = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED)
    if raw_d_u16 is None or rgb_bgr is None:
        return None
    raw_depth_m = raw_d_u16.astype(np.float32) / 1000.0

    per_adapter: Dict[str, Optional[np.ndarray]] = {}
    for name in adapters:
        npy = session_root / "outputs" / name / "depth" / f"{frame_idx:06d}.npy"
        per_adapter[name] = _load_depth(npy)

    grid = make_compare_image(rgb_bgr, raw_depth_m, per_adapter, vmin=vmin, vmax=vmax)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_idx:06d}.png"
    cv2.imwrite(str(out_path), grid)
    return out_path
