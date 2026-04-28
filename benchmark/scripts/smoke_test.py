"""Offline smoke test that exercises the end-to-end harness without a D435i.

It synthesizes a 10-frame "session" on disk consisting of:
  * an RGB checkerboard
  * an IR stereo pair (left = pattern, right = same pattern shifted by a
    constant disparity)
  * a synthetic depth map computed from that disparity (with mild noise)
  * a manifest with reasonable D435i-like intrinsics

Then it runs the ``realsense_hw`` adapter (which doesn't need any GPU
model) followed by the evaluator. The output should be a populated
``results.md`` and a small set of comparison PNGs, and the script asserts
that all the expected files appeared.

Run via::

    python benchmark/scripts/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".smoke_deps"))

from benchmark.io.session import (  # noqa: E402
    CaptureMeta,
    Extrinsics,
    Manifest,
    SessionWriter,
    StreamIntrinsics,
)


def make_intrinsics(w: int, h: int, fx: float, fy: float) -> StreamIntrinsics:
    return StreamIntrinsics(
        width=w, height=h, fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0,
        distortion_model="brown_conrady",
        distortion_coeffs=[0.0] * 5,
    )


def make_synthetic_frame(W: int, H: int, t: int):
    """Returns ir_left, ir_right, color_bgr, depth_mm_u16 for frame index ``t``."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pattern = ((xx // 32) + (yy // 32)) % 2 * 200 + 30
    pattern = pattern.astype(np.uint8)
    drift = (3 * np.sin(t / 5.0)).astype(np.float32)

    ir_left = pattern
    disparity = 30.0 + 0.0 * yy
    shift = int(round(disparity[0, 0]))
    ir_right = np.roll(pattern, -shift, axis=1)

    color_bgr = np.stack([pattern, pattern, pattern], axis=-1)

    fx = 600.0
    baseline_m = 0.05
    depth_m = (fx * baseline_m) / np.maximum(disparity, 1e-3)
    depth_m += np.random.default_rng(t).normal(0.0, 0.005, size=depth_m.shape)
    depth_m += float(drift) * 0.001
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

    return ir_left, ir_right, color_bgr, depth_mm


def build_session(root: Path, n_frames: int = 10) -> None:
    W_ir, H_ir = 320, 240
    W_color, H_color = 320, 240
    fx, fy = 600.0, 600.0

    streams = {
        "ir_left": make_intrinsics(W_ir, H_ir, fx, fy),
        "ir_right": make_intrinsics(W_ir, H_ir, fx, fy),
        "color": make_intrinsics(W_color, H_color, fx, fy),
    }
    extrinsics = {
        "ir_left_to_ir_right": Extrinsics(
            rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            translation=[-0.05, 0.0, 0.0],
        ),
        "ir_left_to_color": Extrinsics(
            rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            translation=[0.015, 0.0, 0.0],
        ),
    }

    capture = CaptureMeta(
        width=W_color, height=H_color, fps=30,
        projector_on=False, duration_s=float(n_frames) / 30.0,
        n_frames=0, depth_scale_to_meters=1.0 / 1000.0,
    )
    manifest = Manifest(
        capture=capture, streams=streams, extrinsics=extrinsics,
        notes="synthetic smoke test session",
    )

    with SessionWriter(root) as w:
        for t in range(n_frames):
            ir_l, ir_r, color, depth = make_synthetic_frame(W_color, H_color, t)
            w.write_frame(t, ir_l, ir_r, color, depth, (0, 0, 0, 0))
        w.write_manifest(manifest)


def run_realsense_hw(session_root: Path) -> None:
    from benchmark.adapters.realsense_hw import RealSenseHWAdapter
    from benchmark.io.results import FrameTiming, OutputWriter, RunMeta
    from benchmark.io.session import SessionReader

    adapter = RealSenseHWAdapter()
    adapter.load({}, "cpu")
    session = SessionReader(session_root)

    writer = OutputWriter(session_root, adapter.name, save_disp=False, save_conf=False)
    writer.write_meta(RunMeta(adapter=adapter.name, target_frame=adapter.target_frame))
    try:
        for fi in session.frame_indices:
            frame = session.frame(fi)
            out = adapter.infer(frame)
            writer.write_depth(fi, out["depth_m"])
            writer.write_timing(FrameTiming(frame_idx=fi, infer_ms=0.5, total_ms=0.5))
    finally:
        writer.close()


def run_evaluator(session_root: Path) -> int:
    from benchmark.runners.evaluate import evaluate_session
    cfg_path = ROOT / "configs" / "default.yaml"
    return evaluate_session(session_root, cfg_path)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bench_smoke_"))
    print(f"[smoke] working dir: {tmp}")
    try:
        session_root = tmp / "smoke_session"
        build_session(session_root, n_frames=12)
        assert (session_root / "manifest.json").exists()
        assert (session_root / "ir_left" / "000000.png").exists()

        run_realsense_hw(session_root)
        assert (session_root / "outputs" / "realsense_hw" / "depth" / "000000.npy").exists()
        assert (session_root / "outputs" / "realsense_hw" / "timings.csv").exists()
        assert (session_root / "outputs" / "realsense_hw" / "meta.json").exists()

        rc = run_evaluator(session_root)
        assert rc == 0, f"evaluator returned non-zero rc={rc}"
        assert (session_root / "results.md").exists()
        assert (session_root / "results.csv").exists()
        compare_dir = session_root / "outputs" / "_compare"
        compare_pngs = list(compare_dir.glob("*.png")) if compare_dir.exists() else []
        assert compare_pngs, "no comparison PNGs were generated"

        print()
        print("[smoke] OK - session built, adapter ran, evaluator produced:")
        print(f"  - {session_root / 'results.md'}")
        print(f"  - {session_root / 'results.csv'}")
        print(f"  - {len(compare_pngs)} compare PNGs")
        return 0
    finally:
        if os.environ.get("BENCH_SMOKE_KEEP") != "1":
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
