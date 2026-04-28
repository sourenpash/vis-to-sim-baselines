"""Per-adapter output writer.

Each invocation of ``runners/run_model.py`` constructs one
:class:`OutputWriter` rooted at ``sessions/<name>/outputs/<adapter>/`` and
populates ``depth/``, ``timings.csv``, ``meta.json``, and (for stereo
models) ``disp/`` / ``conf/``.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


TIMINGS_HEADER = (
    "frame_idx",
    "prep_ms",
    "infer_ms",
    "post_ms",
    "total_ms",
    "gpu_mem_mb",
)


@dataclass
class FrameTiming:
    frame_idx: int
    prep_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0
    total_ms: float = 0.0
    gpu_mem_mb: float = 0.0

    def row(self) -> List:
        return [
            self.frame_idx,
            self.prep_ms,
            self.infer_ms,
            self.post_ms,
            self.total_ms,
            self.gpu_mem_mb,
        ]


@dataclass
class RunMeta:
    """Provenance and environment fingerprint for one model run."""
    adapter: str
    target_frame: str
    conda_env: Optional[str] = None
    python_version: Optional[str] = None
    torch_version: Optional[str] = None
    cuda_available: Optional[bool] = None
    gpu_name: Optional[str] = None
    weights_path: Optional[str] = None
    weights_sha256: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutputWriter:
    """Streaming writer for ``sessions/<name>/outputs/<adapter>/``."""

    def __init__(
        self,
        session_root: os.PathLike,
        adapter_name: str,
        save_disp: bool = True,
        save_conf: bool = False,
    ) -> None:
        self.root = Path(session_root) / "outputs" / adapter_name
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "depth").mkdir(exist_ok=True)
        self.save_disp = save_disp
        self.save_conf = save_conf
        if save_disp:
            (self.root / "disp").mkdir(exist_ok=True)
        if save_conf:
            (self.root / "conf").mkdir(exist_ok=True)

        self._timings_path = self.root / "timings.csv"
        self._timings_file = open(self._timings_path, "w", newline="")
        self._timings_writer = csv.writer(self._timings_file)
        self._timings_writer.writerow(TIMINGS_HEADER)
        self._timings_file.flush()

    def write_depth(self, idx: int, depth_m: np.ndarray) -> None:
        if depth_m.dtype != np.float32:
            depth_m = depth_m.astype(np.float32)
        np.save(self.root / "depth" / f"{idx:06d}.npy", depth_m)

    def write_disparity(self, idx: int, disp_px: np.ndarray) -> None:
        if not self.save_disp:
            return
        if disp_px.dtype != np.float32:
            disp_px = disp_px.astype(np.float32)
        np.save(self.root / "disp" / f"{idx:06d}.npy", disp_px)

    def write_confidence(self, idx: int, conf: np.ndarray) -> None:
        if not self.save_conf:
            return
        if conf.dtype != np.float32:
            conf = conf.astype(np.float32)
        np.save(self.root / "conf" / f"{idx:06d}.npy", conf)

    def write_timing(self, t: FrameTiming) -> None:
        self._timings_writer.writerow(t.row())
        self._timings_file.flush()

    def write_meta(self, meta: RunMeta) -> None:
        with open(self.root / "meta.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

    def close(self) -> None:
        try:
            self._timings_file.close()
        except Exception:
            pass

    def __enter__(self) -> "OutputWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
