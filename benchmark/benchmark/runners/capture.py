"""Record a synchronized D435i session to disk.

Usage::

    python -m benchmark.runners.capture --out sessions/my_run --duration 30

This writes ``manifest.json``, four PNG streams (ir_left, ir_right, color,
depth) and ``timestamps.csv`` under ``--out``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from ..camera.d435i import D435iCapture, D435iConfig
from ..io.session import SessionWriter


def _load_capture_cfg(cfg_path: Optional[Path]) -> D435iConfig:
    if cfg_path is None:
        return D435iConfig()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cap = cfg.get("capture", {})
    return D435iConfig(
        ir_width=cap.get("ir_width", 848),
        ir_height=cap.get("ir_height", 480),
        ir_fps=cap.get("ir_fps", 30),
        color_width=cap.get("color_width", 640),
        color_height=cap.get("color_height", 480),
        color_fps=cap.get("color_fps", 30),
        depth_width=cap.get("depth_width", 848),
        depth_height=cap.get("depth_height", 480),
        depth_fps=cap.get("depth_fps", 30),
        align_depth_to=cap.get("align_depth_to", "color"),
        emitter_enabled=bool(cap.get("emitter_enabled", False)),
    )


def _warmup_frames(cfg_path: Optional[Path]) -> int:
    if cfg_path is None:
        return 30
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return int(cfg.get("capture", {}).get("warmup_frames", 30))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.runners.capture",
        description="Record a synchronized D435i session.",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Session directory to create.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a YAML config (default: benchmark/configs/default.yaml).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Capture duration in seconds (overrides config.capture.duration_s).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Hard cap on number of frames recorded.",
    )
    parser.add_argument(
        "--emitter", choices=["on", "off"], default=None,
        help="Override IR projector (default from config).",
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Free-form notes stored in manifest.json.",
    )
    args = parser.parse_args(argv)

    cfg_path = args.config
    if cfg_path is None:
        default = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        if default.exists():
            cfg_path = default

    cap_cfg = _load_capture_cfg(cfg_path)
    warmup = _warmup_frames(cfg_path)

    if args.emitter is not None:
        cap_cfg.emitter_enabled = (args.emitter == "on")

    duration = args.duration
    if duration is None and cfg_path is not None:
        with open(cfg_path, "r") as f:
            duration = float(((yaml.safe_load(f) or {}).get("capture", {})).get("duration_s", 30.0))
    if duration is None:
        duration = 30.0

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[capture] target dir   : {args.out}")
    print(f"[capture] duration     : {duration:.1f}s")
    print(f"[capture] emitter      : {'ON' if cap_cfg.emitter_enabled else 'OFF'}")
    print(f"[capture] ir   stream  : {cap_cfg.ir_width}x{cap_cfg.ir_height}@{cap_cfg.ir_fps}")
    print(f"[capture] color stream : {cap_cfg.color_width}x{cap_cfg.color_height}@{cap_cfg.color_fps}")

    writer = SessionWriter(args.out)
    n_written = 0
    t_first: Optional[float] = None
    try:
        cam = D435iCapture(cap_cfg)
        cam.start()
        try:
            print(f"[capture] warming up {warmup} frames ...")
            for _ in range(warmup):
                cam.read_one()
            print("[capture] recording ...")
            t_first = time.perf_counter()
            for f in cam.stream(warmup_frames=0, max_frames=args.max_frames):
                writer.write_frame(
                    f.idx,
                    f.ir_left, f.ir_right, f.color_bgr, f.depth_mm_u16,
                    (f.host_t_ns, f.ir_t_ns, f.color_t_ns, f.depth_t_ns),
                )
                n_written = f.idx + 1
                elapsed = time.perf_counter() - t_first
                if elapsed >= duration:
                    break
                if n_written % 30 == 0:
                    print(f"  frame {n_written:5d}  elapsed {elapsed:6.2f}s")
        finally:
            elapsed = (time.perf_counter() - t_first) if t_first else 0.0
            manifest = cam.build_manifest(
                duration_s=elapsed,
                n_frames=n_written,
                notes=args.notes,
            )
            writer.write_manifest(manifest)
            cam.stop()
    finally:
        writer.close()

    print(f"[capture] done: {n_written} frames in {elapsed:.2f}s "
          f"({(n_written / max(elapsed, 1e-6)):.1f} FPS) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
