"""Run one depth adapter on a recorded session or live D435i feed.

Replay mode iterates over a session's frames in order, feeding each one
into the adapter and writing per-frame outputs + timings to
``sessions/<name>/outputs/<adapter>/``.

Live mode opens the camera in-process and streams frames into the adapter
without writing PNGs; only ``timings.csv`` and a small number of sample
``depth/*.npy`` files are saved.

This script is invoked once per env::

    conda run -n ffs            python -m benchmark.runners.run_model --adapter ffs ...
    conda run -n s2m2           python -m benchmark.runners.run_model --adapter s2m2 ...
    conda run -n lingbot-depth  python -m benchmark.runners.run_model --adapter lingbot ...
    conda run -n bench          python -m benchmark.runners.run_model --adapter realsense_hw ...
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import yaml

from ..adapters.base import DepthAdapter
from ..io.results import FrameTiming, OutputWriter, RunMeta
from ..io.session import Manifest, SessionReader


ADAPTER_REGISTRY = {
    "realsense_hw": ("benchmark.adapters.realsense_hw", "RealSenseHWAdapter"),
    "ffs":          ("benchmark.adapters.ffs", "FFSAdapter"),
    "s2m2":         ("benchmark.adapters.s2m2", "S2M2Adapter"),
    "lingbot":      ("benchmark.adapters.lingbot", "LingBotAdapter"),
}


def _import_adapter(name: str) -> DepthAdapter:
    if name not in ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown adapter '{name}'. Known: {sorted(ADAPTER_REGISTRY)}"
        )
    mod_path, cls_name = ADAPTER_REGISTRY[name]
    import importlib
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls()


class _Timer:
    """Synchronous timer that uses cuda events when available."""

    def __init__(self, device: str) -> None:
        self.use_cuda = device.startswith("cuda")
        if self.use_cuda:
            import torch
            self._torch = torch
            self._sync = torch.cuda.synchronize
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.use_cuda:
            self._sync()
            self.start_ev.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_cuda:
            self.end_ev.record()
            self._sync()
            self.elapsed_ms = self.start_ev.elapsed_time(self.end_ev)
        else:
            self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0


def _gpu_mem_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _reset_gpu_mem() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _build_run_meta(
    adapter: DepthAdapter,
    cfg: Dict[str, Any],
    target_frame: str,
) -> RunMeta:
    meta = RunMeta(
        adapter=adapter.name,
        target_frame=target_frame,
        conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
        python_version=platform.python_version(),
        config=cfg,
    )
    try:
        import torch
        meta.torch_version = torch.__version__
        meta.cuda_available = bool(torch.cuda.is_available())
        if meta.cuda_available:
            meta.gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    meta.extra = adapter.meta()
    return meta


def _frames_from_session(
    session: SessionReader,
    max_frames: Optional[int],
) -> Iterator[Dict[str, Any]]:
    n = len(session)
    if max_frames is not None:
        n = min(n, max_frames)
    for i, fi in enumerate(session.frame_indices[:n]):
        yield session.frame(fi)


def _frames_from_camera(
    cap_cfg_path: Optional[Path],
    sample_session: Optional[SessionReader],
    max_frames: int,
) -> Iterator[Dict[str, Any]]:
    """Live D435i frames packaged as adapter ``frame`` dicts.

    If ``sample_session`` is provided, intrinsics and the baseline are read
    from its manifest (i.e. captured beforehand by ``runners/capture.py``);
    otherwise we read them off the running camera stream.
    """
    from ..camera.d435i import D435iCapture, D435iConfig

    cap_cfg = D435iConfig()
    if cap_cfg_path is not None and cap_cfg_path.exists():
        with open(cap_cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        cap = cfg.get("capture", {})
        for k, default in cap_cfg.__dict__.items():
            if k in cap:
                setattr(cap_cfg, k, type(default)(cap[k]))

    cam = D435iCapture(cap_cfg)
    cam.start()
    try:
        if sample_session is not None:
            manifest = sample_session.manifest
        else:
            manifest = cam.build_manifest(duration_s=0.0, n_frames=0)
        intrinsics = {k: v.K() for k, v in manifest.streams.items()}
        baseline_m = manifest.stereo_baseline_m
        depth_scale_m = manifest.capture.depth_scale_to_meters

        i = 0
        for cf in cam.stream(warmup_frames=30, max_frames=max_frames):
            depth_m = cf.depth_mm_u16.astype(np.float32) * float(depth_scale_m)
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            import cv2
            color_rgb = cv2.cvtColor(cf.color_bgr, cv2.COLOR_BGR2RGB)
            yield {
                "frame_idx": i,
                "ir_left": cf.ir_left,
                "ir_right": cf.ir_right,
                "color": color_rgb,
                "raw_depth_m": depth_m,
                "intrinsics": intrinsics,
                "baseline_m": baseline_m,
                "manifest": manifest,
            }
            i += 1
    finally:
        cam.stop()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.runners.run_model",
        description="Run one depth adapter on a session (replay) or live D435i.",
    )
    parser.add_argument("--adapter", required=True, choices=list(ADAPTER_REGISTRY.keys()))
    parser.add_argument("--session", required=True, type=Path,
                        help="Session directory (read for replay, written to in both modes).")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to YAML config (default: benchmark/configs/default.yaml).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--live", action="store_true",
                        help="Stream from D435i instead of reading session frames.")
    parser.add_argument("--live-samples", type=int, default=10,
                        help="In live mode, save the first N depth maps as samples.")
    parser.add_argument("--no-save-disp", action="store_true")
    parser.add_argument("--no-save-depth", action="store_true",
                        help="Skip writing per-frame .npy files (keep only timings).")
    args = parser.parse_args(argv)

    cfg_path = args.config
    if cfg_path is None:
        default = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        if default.exists():
            cfg_path = default
    if cfg_path is None or not cfg_path.exists():
        cfg_all = {"models": {}, "run": {}}
    else:
        with open(cfg_path, "r") as f:
            cfg_all = yaml.safe_load(f) or {}
    model_cfg = (cfg_all.get("models", {}) or {}).get(args.adapter, {}) or {}

    if not args.live:
        if not args.session.exists():
            print(f"[run_model] session not found: {args.session}", file=sys.stderr)
            return 2
        session = SessionReader(args.session)
    else:
        args.session.mkdir(parents=True, exist_ok=True)
        manifest_path = args.session / "manifest.json"
        session = SessionReader(args.session) if manifest_path.exists() else None

    adapter = _import_adapter(args.adapter)
    print(f"[run_model] adapter={adapter.name} device={args.device} live={args.live}")
    print(f"[run_model] config: {model_cfg}")

    adapter.load(model_cfg, args.device)

    target_frame = adapter.target_frame

    save_disp = (not args.no_save_disp) and bool(cfg_all.get("run", {}).get("save_disp", True))
    save_conf = bool(cfg_all.get("run", {}).get("save_conf", False))

    writer = OutputWriter(args.session, adapter.name, save_disp=save_disp, save_conf=save_conf)
    try:
        if args.warmup > 0:
            if not args.live and session is not None and len(session) > 0:
                first = session.frame(session.frame_indices[0])
                print(f"[run_model] warming up {args.warmup} iters on frame 0 ...")
                adapter.warmup(first, n=args.warmup)
            else:
                print("[run_model] live mode: skipping pre-warmup; first frame timings may be slow")

        run_meta = _build_run_meta(adapter, model_cfg, target_frame)
        writer.write_meta(run_meta)

        max_frames = args.max_frames
        if max_frames is None:
            cfg_max = cfg_all.get("run", {}).get("max_frames")
            max_frames = int(cfg_max) if cfg_max not in (None, "null") else None

        if args.live:
            frames_iter = _frames_from_camera(cfg_path, session, max_frames or 600)
        else:
            frames_iter = _frames_from_session(session, max_frames)

        wall_t0 = time.perf_counter()
        n = 0
        last_log = wall_t0
        for frame in frames_iter:
            _reset_gpu_mem()
            with _Timer(args.device) as total_t:
                with _Timer(args.device) as infer_t:
                    out = adapter.infer(frame)

            depth_m = out["depth_m"]
            disp = out.get("disparity_px")
            conf = out.get("confidence")

            if not args.no_save_depth:
                if args.live and n >= args.live_samples:
                    pass
                else:
                    writer.write_depth(frame["frame_idx"], depth_m)
                    if save_disp and disp is not None:
                        writer.write_disparity(frame["frame_idx"], disp)
                    if save_conf and conf is not None:
                        writer.write_confidence(frame["frame_idx"], conf)

            timing = FrameTiming(
                frame_idx=int(frame["frame_idx"]),
                prep_ms=0.0,
                infer_ms=float(infer_t.elapsed_ms),
                post_ms=0.0,
                total_ms=float(total_t.elapsed_ms),
                gpu_mem_mb=float(_gpu_mem_mb()),
            )
            writer.write_timing(timing)

            n += 1
            now = time.perf_counter()
            if now - last_log >= 2.0:
                last_log = now
                print(
                    f"  frame {n:4d}  infer {timing.infer_ms:6.1f} ms  "
                    f"total {timing.total_ms:6.1f} ms  gpu {timing.gpu_mem_mb:.0f} MB"
                )

        wall = time.perf_counter() - wall_t0
        if n > 0:
            print(
                f"[run_model] done: {n} frames in {wall:.2f}s "
                f"(wall FPS {(n / max(wall, 1e-6)):.1f})"
            )
        else:
            print("[run_model] no frames processed")
    finally:
        adapter.close()
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
