"""Adapter for Fast-FoundationStereo (env ``ffs``).

Loads a serialized ``FastFoundationStereo`` model from a ``.pth`` file (the
upstream weights are pickled full-model objects, not a state_dict), tiles
left/right IR images to 3 channels, pads to a multiple of 32, and runs a
single ``forward(..., test_mode=True)`` pass. Disparity is converted to
metric depth via ``fx * baseline / disp`` from the session manifest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import DepthAdapter


def _add_repo_to_path(repo_root: Path) -> None:
    """Make ``import core.foundation_stereo`` work without installing the repo."""
    repo_root = Path(repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class FFSAdapter(DepthAdapter):
    name = "ffs"
    target_frame = "ir_left"

    def __init__(self) -> None:
        self._model = None
        self._padder_cls = None
        self._device = None
        self._weights_path: str = ""
        self._cfg: Dict[str, Any] = {}

    def load(self, cfg: Dict[str, Any], device: str) -> None:
        import torch

        repo_root = Path(cfg.get("repo_root", "../Fast-FoundationStereo"))
        if not repo_root.is_absolute():
            repo_root = (Path(__file__).resolve().parents[2] / repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"FFS repo not found: {repo_root}")
        _add_repo_to_path(repo_root)

        from core.utils.utils import InputPadder
        self._padder_cls = InputPadder

        weights_rel = cfg.get("weights", "weights/23-36-37/model_best_bp2_serialize.pth")
        weights_path = (repo_root / weights_rel).resolve()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"FFS weights not found at {weights_path}. Download per FFS README."
            )
        self._weights_path = str(weights_path)

        torch.autograd.set_grad_enabled(False)
        model = torch.load(str(weights_path), map_location="cpu", weights_only=False)

        valid_iters = int(cfg.get("valid_iters", 8))
        max_disp = int(cfg.get("max_disp", 192))
        if hasattr(model, "args"):
            try:
                model.args.valid_iters = valid_iters
                model.args.max_disp = max_disp
            except Exception:
                pass

        self._device = torch.device(device)
        model = model.to(self._device).eval()
        self._model = model
        self._cfg = dict(cfg)
        self._valid_iters = valid_iters
        self._max_disp = max_disp
        self._optimize_build_volume = cfg.get("optimize_build_volume", "pytorch1")
        self._fp16 = bool(cfg.get("fp16", True))

    def _ir_to_tensor(self, ir: np.ndarray):
        import torch
        if ir.ndim == 2:
            ir3 = np.repeat(ir[..., None], 3, axis=-1)
        else:
            ir3 = ir[..., :3]
        ir3 = ir3.astype(np.float32, copy=False)
        t = torch.from_numpy(ir3).to(self._device).float()
        return t.permute(2, 0, 1).unsqueeze(0)

    def infer(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        assert self._model is not None, "FFSAdapter.load() must be called first"

        left = self._ir_to_tensor(frame["ir_left"])
        right = self._ir_to_tensor(frame["ir_right"])

        H, W = left.shape[-2:]
        padder = self._padder_cls(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)

        amp_dtype = torch.float16 if self._fp16 else torch.float32
        with torch.amp.autocast("cuda", enabled=self._fp16, dtype=amp_dtype):
            disp = self._model.forward(
                left, right,
                iters=self._valid_iters,
                test_mode=True,
                optimize_build_volume=self._optimize_build_volume,
            )
        disp = padder.unpad(disp.float())
        disp_np = disp.detach().cpu().numpy().reshape(H, W).clip(0.0, None)

        K_ir = frame["intrinsics"]["ir_left"]
        fx = float(K_ir[0, 0])
        baseline_m = float(frame["baseline_m"])

        eps = 1e-6
        depth_m = (fx * baseline_m) / np.maximum(disp_np, eps)
        depth_m = depth_m.astype(np.float32, copy=False)
        depth_m[disp_np <= eps] = 0.0
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "depth_m": depth_m,
            "disparity_px": disp_np.astype(np.float32, copy=False),
        }

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
            "weights_path": self._weights_path,
            "valid_iters": getattr(self, "_valid_iters", None),
            "max_disp": getattr(self, "_max_disp", None),
            "optimize_build_volume": getattr(self, "_optimize_build_volume", None),
            "fp16": getattr(self, "_fp16", None),
        }
