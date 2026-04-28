"""Session directory I/O.

A "session" is a self-contained, env-agnostic snapshot of synchronized D435i
streams plus calibration. Every adapter reads a session via :class:`SessionReader`
no matter which conda env it runs in; the only required deps are numpy + opencv.

Layout::

    sessions/<name>/
      manifest.json
      ir_left/<frame_idx:06d>.png       8-bit, rectified
      ir_right/<frame_idx:06d>.png      8-bit, rectified
      color/<frame_idx:06d>.png         8-bit BGR (cv2.imwrite default)
      depth/<frame_idx:06d>.png         16-bit, mm, aligned to color
      timestamps.csv
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


MANIFEST_NAME = "manifest.json"
TIMESTAMPS_NAME = "timestamps.csv"
STREAM_DIRS = ("ir_left", "ir_right", "color", "depth")


@dataclass
class StreamIntrinsics:
    """Pinhole intrinsics for one camera stream."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "brown_conrady"
    distortion_coeffs: List[float] = field(default_factory=lambda: [0.0] * 5)

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass
class Extrinsics:
    """3x3 rotation + 3x1 translation, both row-major flat lists for JSON."""
    rotation: List[float]
    translation: List[float]

    def R(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=np.float32).reshape(3, 3)

    def t(self) -> np.ndarray:
        return np.asarray(self.translation, dtype=np.float32).reshape(3)


@dataclass
class CaptureMeta:
    width: int
    height: int
    fps: int
    projector_on: bool
    duration_s: float
    n_frames: int
    depth_scale_to_meters: float = 1.0 / 1000.0


@dataclass
class Manifest:
    """Top-level session metadata stored as ``manifest.json``."""
    capture: CaptureMeta
    streams: Dict[str, StreamIntrinsics]
    extrinsics: Dict[str, Extrinsics]
    rois: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capture": asdict(self.capture),
            "streams": {k: asdict(v) for k, v in self.streams.items()},
            "extrinsics": {k: asdict(v) for k, v in self.extrinsics.items()},
            "rois": self.rois,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Manifest":
        return cls(
            capture=CaptureMeta(**d["capture"]),
            streams={k: StreamIntrinsics(**v) for k, v in d["streams"].items()},
            extrinsics={k: Extrinsics(**v) for k, v in d["extrinsics"].items()},
            rois=d.get("rois", {}),
            notes=d.get("notes"),
            schema_version=d.get("schema_version", 1),
        )

    @property
    def stereo_baseline_m(self) -> float:
        """Magnitude of the IR_left -> IR_right translation, in meters."""
        if "ir_left_to_ir_right" not in self.extrinsics:
            raise KeyError("ir_left_to_ir_right not present in extrinsics")
        t = self.extrinsics["ir_left_to_ir_right"].t()
        return float(np.linalg.norm(t))


class SessionWriter:
    """Streaming writer used by ``runners/capture.py``."""

    def __init__(self, root: os.PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for d in STREAM_DIRS:
            (self.root / d).mkdir(exist_ok=True)
        self._ts_path = self.root / TIMESTAMPS_NAME
        self._ts_file = open(self._ts_path, "w", newline="")
        self._ts_writer = csv.writer(self._ts_file)
        self._ts_writer.writerow(
            ["frame_idx", "host_t_ns", "ir_t_ns", "color_t_ns", "depth_t_ns"]
        )
        self._ts_file.flush()
        self._n = 0

    def write_frame(
        self,
        idx: int,
        ir_left: np.ndarray,
        ir_right: np.ndarray,
        color_bgr: np.ndarray,
        depth_mm_u16: np.ndarray,
        timestamps_ns: Tuple[int, int, int, int],
    ) -> None:
        """Persist one frame across all four streams."""
        name = f"{idx:06d}.png"
        cv2.imwrite(str(self.root / "ir_left" / name), ir_left)
        cv2.imwrite(str(self.root / "ir_right" / name), ir_right)
        cv2.imwrite(str(self.root / "color" / name), color_bgr)
        if depth_mm_u16.dtype != np.uint16:
            depth_mm_u16 = depth_mm_u16.astype(np.uint16)
        cv2.imwrite(str(self.root / "depth" / name), depth_mm_u16)
        self._ts_writer.writerow([idx, *timestamps_ns])
        self._ts_file.flush()
        self._n = max(self._n, idx + 1)

    def write_manifest(self, manifest: Manifest) -> None:
        manifest.capture.n_frames = self._n
        with open(self.root / MANIFEST_NAME, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def close(self) -> None:
        try:
            self._ts_file.close()
        except Exception:
            pass

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SessionReader:
    """Random-access reader for a session.

    Reads from disk lazily per ``__getitem__``; safe to use across processes.
    """

    def __init__(self, root: os.PathLike) -> None:
        self.root = Path(root)
        if not (self.root / MANIFEST_NAME).exists():
            raise FileNotFoundError(f"Missing {MANIFEST_NAME} in {self.root}")
        with open(self.root / MANIFEST_NAME, "r") as f:
            self.manifest = Manifest.from_dict(json.load(f))
        self._frames = sorted(
            int(p.stem) for p in (self.root / "ir_left").glob("*.png")
        )

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frame_indices(self) -> List[int]:
        return list(self._frames)

    def _png(self, sub: str, idx: int, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
        p = self.root / sub / f"{idx:06d}.png"
        img = cv2.imread(str(p), flags)
        if img is None:
            raise FileNotFoundError(f"Cannot read {p}")
        return img

    def read_ir_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 1-channel uint8 left/right IR images."""
        left = self._png("ir_left", idx, cv2.IMREAD_UNCHANGED)
        right = self._png("ir_right", idx, cv2.IMREAD_UNCHANGED)
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if right.ndim == 3:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return left, right

    def read_color_rgb(self, idx: int) -> np.ndarray:
        """Returns RGB uint8 ``H x W x 3``."""
        bgr = self._png("color", idx, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read_color_bgr(self, idx: int) -> np.ndarray:
        return self._png("color", idx, cv2.IMREAD_COLOR)

    def read_depth_m(self, idx: int) -> np.ndarray:
        """Returns float32 depth in meters."""
        d_u16 = self._png("depth", idx, cv2.IMREAD_UNCHANGED)
        if d_u16.dtype != np.uint16:
            d_u16 = d_u16.astype(np.uint16)
        depth_m = d_u16.astype(np.float32) * float(self.manifest.capture.depth_scale_to_meters)
        return np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    def frame(self, idx: int) -> Dict[str, Any]:
        """Returns the dict consumed by :meth:`DepthAdapter.infer`."""
        ir_left, ir_right = self.read_ir_pair(idx)
        color_rgb = self.read_color_rgb(idx)
        depth_m = self.read_depth_m(idx)
        return {
            "frame_idx": idx,
            "ir_left": ir_left,
            "ir_right": ir_right,
            "color": color_rgb,
            "raw_depth_m": depth_m,
            "intrinsics": {k: v.K() for k, v in self.manifest.streams.items()},
            "baseline_m": self.manifest.stereo_baseline_m,
            "manifest": self.manifest,
        }

    def iter_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        end = len(self) if end is None else min(end, len(self))
        for i in range(start, end):
            yield self.frame(self._frames[i])
