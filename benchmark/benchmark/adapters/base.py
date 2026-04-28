"""Abstract base class shared by all depth adapters."""

from __future__ import annotations

import abc
from typing import Any, Dict


class DepthAdapter(abc.ABC):
    """Uniform interface over heterogeneous depth/stereo models.

    Every concrete adapter lives in its own conda env (because the upstream
    repos pin incompatible torch/xformers stacks), but obeys this shared
    contract so the runners and evaluator never need model-specific code.

    Attributes
    ----------
    name : str
        Short identifier, used as the output subdirectory name.
    target_frame : str
        Which frame's intrinsics the returned depth map is aligned to.
        ``"ir_left"`` for stereo matchers, ``"color"`` for monocular /
        refinement models.
    """

    name: str = "base"
    target_frame: str = "ir_left"

    @abc.abstractmethod
    def load(self, cfg: Dict[str, Any], device: str) -> None:
        """Load weights and move the model to ``device`` (e.g. ``"cuda"``)."""

    def warmup(self, frame: Dict[str, Any], n: int = 3) -> None:
        """Run ``infer`` ``n`` times on ``frame`` to fill caches / compile.

        Default implementation is a simple loop; adapters can override if a
        cheaper warmup path exists.
        """
        for _ in range(max(0, int(n))):
            self.infer(frame)

    @abc.abstractmethod
    def infer(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference for a single frame.

        Parameters
        ----------
        frame : dict
            Provided by :class:`benchmark.io.session.SessionReader.frame`.
            Keys: ``ir_left``, ``ir_right``, ``color``, ``raw_depth_m``,
            ``intrinsics`` (dict of stream name -> 3x3 float32 K),
            ``baseline_m``, ``manifest``, ``frame_idx``.

        Returns
        -------
        dict
            Must contain ``depth_m`` (HxW float32 numpy array, meters,
            spatially aligned with ``self.target_frame``). May optionally
            contain ``disparity_px`` (HxW float32) for stereo models and
            ``confidence`` (HxW float32 in [0, 1]).
        """

    def close(self) -> None:
        """Release GPU memory etc. Called once at the end of a run."""
        return None

    def meta(self) -> Dict[str, Any]:
        """Optional adapter-specific metadata for ``meta.json``."""
        return {}
