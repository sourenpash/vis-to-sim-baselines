"""Adapter for LingBot-Depth (env ``lingbot-depth``).

Loads ``MDMModel`` from a Hugging Face repo id (or local checkpoint) and
runs depth refinement using the D435i color frame plus the D435i hardware
depth aligned to color. Intrinsics must be passed in normalized form
(``fx/W``, ``fy/H``, ``cx/W``, ``cy/H``).

Note: this model is a depth refiner. ``forward`` asserts ``depth is not
None``, so we always pass the raw aligned depth from the session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import DepthAdapter


def _add_lingbot_to_path(repo_root: Path) -> None:
    """Allow ``import mdm.model.v2`` without installing the package."""
    repo_root = Path(repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class LingBotAdapter(DepthAdapter):
    name = "lingbot"
    target_frame = "color"

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._cfg: Dict[str, Any] = {}
        self._weights_id: str = ""

    def load(self, cfg: Dict[str, Any], device: str) -> None:
        import torch

        repo_root = Path(cfg.get("repo_root", "../lingbot-depth"))
        if not repo_root.is_absolute():
            repo_root = (Path(__file__).resolve().parents[2] / repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"lingbot-depth repo not found: {repo_root}")
        _add_lingbot_to_path(repo_root)

        from mdm.model.v2 import MDMModel

        weights = cfg.get("hf_repo_id") or cfg.get("weights")
        if not weights:
            weights = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
        self._weights_id = str(weights)

        self._device = torch.device(device)
        model = MDMModel.from_pretrained(weights).to(self._device)
        model.eval()
        self._model = model
        self._cfg = dict(cfg)
        self._resolution_level = int(cfg.get("resolution_level", 9))
        self._use_fp16 = bool(cfg.get("use_fp16", True))
        self._apply_mask = bool(cfg.get("apply_mask", True))

    def _normalized_K(self, K: np.ndarray, W: int, H: int) -> np.ndarray:
        K_norm = K.astype(np.float32, copy=True)
        K_norm[0, 0] /= W
        K_norm[0, 2] /= W
        K_norm[1, 1] /= H
        K_norm[1, 2] /= H
        return K_norm

    def infer(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        assert self._model is not None, "LingBotAdapter.load() must be called first"

        color_rgb = frame["color"]
        H, W = color_rgb.shape[:2]
        image_t = (
            torch.from_numpy(color_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1).unsqueeze(0)
            .to(self._device)
        )

        depth_in = frame["raw_depth_m"].astype(np.float32, copy=False)
        if depth_in.shape != (H, W):
            import cv2
            depth_in = cv2.resize(depth_in, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_t = torch.from_numpy(depth_in).unsqueeze(0).to(self._device)

        K = frame["intrinsics"]["color"]
        K_norm = self._normalized_K(K, W, H)
        K_t = torch.from_numpy(K_norm).unsqueeze(0).to(self._device)

        with torch.no_grad():
            out = self._model.infer(
                image_t,
                depth_in=depth_t,
                resolution_level=self._resolution_level,
                use_fp16=self._use_fp16,
                apply_mask=self._apply_mask,
                intrinsics=K_t,
            )

        depth = out["depth"].squeeze().detach().float().cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return {"depth_m": depth}

    def close(self) -> None:
        try:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()
        except Exception:
            pass

    def meta(self) -> Dict[str, Any]:
        return {
            "weights": self._weights_id,
            "resolution_level": getattr(self, "_resolution_level", None),
            "use_fp16": getattr(self, "_use_fp16", None),
            "apply_mask": getattr(self, "_apply_mask", None),
        }
