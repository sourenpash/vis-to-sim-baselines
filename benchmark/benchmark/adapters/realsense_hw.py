"""Pass-through adapter for the D435i hardware depth.

Acts as a non-learned baseline. The hardware depth is already recorded in
the session as ``depth/*.png`` aligned to the color frame, so ``infer``
just reshapes and returns it. Useful for the comparison table and as a
stability/latency floor.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import DepthAdapter


class RealSenseHWAdapter(DepthAdapter):
    name = "realsense_hw"
    target_frame = "color"

    def load(self, cfg: Dict[str, Any], device: str) -> None:  # noqa: ARG002
        return None

    def infer(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        depth_m = frame["raw_depth_m"]
        if depth_m.dtype != np.float32:
            depth_m = depth_m.astype(np.float32)
        return {"depth_m": depth_m}

    def meta(self) -> Dict[str, Any]:
        return {"description": "D435i hardware depth, aligned to color"}
