"""Adapter for S2M2 (env ``s2m2``).

Uses the upstream ``load_model`` helper to instantiate one of S/M/L/XL,
tiles IR frames to 3 channels, then defers to ``run_stereo_matching``
which already handles padding/cropping internally. Disparity is converted
to metric depth via ``fx * baseline / disp`` from the session manifest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import DepthAdapter


def _add_s2m2_to_path(repo_root: Path) -> None:
    """Make ``import s2m2.core...`` resolve when the package isn't installed."""
    repo_root = Path(repo_root).resolve()
    src = repo_root / "src"
    for p in (str(src), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


class S2M2Adapter(DepthAdapter):
    name = "s2m2"
    target_frame = "ir_left"

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._run_stereo_matching = None
        self._weights_dir: str = ""
        self._cfg: Dict[str, Any] = {}
        self._allow_negative = False

    def load(self, cfg: Dict[str, Any], device: str) -> None:
        import torch
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cudnn.benchmark = True

        repo_root = Path(cfg.get("repo_root", "../s2m2"))
        if not repo_root.is_absolute():
            repo_root = (Path(__file__).resolve().parents[2] / repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"s2m2 repo not found: {repo_root}")
        _add_s2m2_to_path(repo_root)

        from s2m2.core.utils.model_utils import load_model, run_stereo_matching

        weights_dir = (
            repo_root / cfg.get("weights_dir", "weights/pretrain_weights")
        ).resolve()
        if not weights_dir.exists():
            raise FileNotFoundError(
                f"s2m2 weights dir not found at {weights_dir}. "
                "Download CHxxNTRy.pth from huggingface (minimok/s2m2)."
            )
        self._weights_dir = str(weights_dir)

        model_type = cfg.get("model_type", "S")
        refine_iter = int(cfg.get("refine_iter", 3))
        use_positivity = bool(cfg.get("use_positivity", True))
        self._allow_negative = not use_positivity

        self._device = torch.device(device)
        self._model = load_model(
            str(weights_dir),
            model_type,
            use_positivity=use_positivity,
            refine_iter=refine_iter,
            device=self._device,
        )
        if self._model is None:
            raise RuntimeError("s2m2 load_model returned None; check checkpoint integrity")

        self._run_stereo_matching = run_stereo_matching
        self._cfg = dict(cfg)
        self._fp16 = bool(cfg.get("fp16", True))
        self._model_type = model_type
        self._refine_iter = refine_iter

    def _ir_to_tensor(self, ir: np.ndarray):
        import torch
        if ir.ndim == 2:
            ir3 = np.repeat(ir[..., None], 3, axis=-1)
        else:
            ir3 = ir[..., :3]
        ir3 = ir3.astype(np.uint8, copy=False)
        t = torch.from_numpy(ir3).permute(2, 0, 1).unsqueeze(0)
        return t.to(self._device)

    def infer(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        assert self._model is not None, "S2M2Adapter.load() must be called first"

        left = self._ir_to_tensor(frame["ir_left"])
        right = self._ir_to_tensor(frame["ir_right"])

        pred_disp, pred_occ, pred_conf, _avg_conf, _run_time = self._run_stereo_matching(
            self._model, left, right, self._device, N_repeat=1,
        )
        disp_np = pred_disp.detach().cpu().numpy().astype(np.float32)
        if disp_np.ndim == 3:
            disp_np = disp_np[0]
        conf_np = pred_conf.detach().cpu().numpy().astype(np.float32)
        if conf_np.ndim == 3:
            conf_np = conf_np[0]

        K_ir = frame["intrinsics"]["ir_left"]
        fx = float(K_ir[0, 0])
        baseline_m = float(frame["baseline_m"])

        eps = 1e-6
        disp_for_depth = np.where(disp_np > eps, disp_np, np.nan)
        depth_m = (fx * baseline_m) / disp_for_depth
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return {
            "depth_m": depth_m,
            "disparity_px": disp_np,
            "confidence": conf_np,
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
            "weights_dir": self._weights_dir,
            "model_type": getattr(self, "_model_type", None),
            "refine_iter": getattr(self, "_refine_iter", None),
            "use_positivity": not self._allow_negative,
            "fp16": getattr(self, "_fp16", None),
        }
