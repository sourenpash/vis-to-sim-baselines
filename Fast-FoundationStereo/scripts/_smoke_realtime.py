"""Offline smoke test for ``realtime_d435i.py``.

Patches the D435i camera + the model load + the inference call so the
script can run end-to-end on a machine without a RealSense camera or
CUDA. Verifies argparse, the per-frame loop, FPS / latency aggregation,
CSV writing, recording, signal handling, and the summary print.

The actual depth network and AprilTag detection are exercised
separately:

* Model load & forward are exercised by ``scripts/run_demo.py`` itself.
* AprilTag detection with a real printed tag36h11 must be verified
  against your actual D435i once the camera is attached.

Run::

    PYTHONPATH=.smoke_deps python scripts/_smoke_realtime.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
sys.path.insert(0, str(REPO_DIR))
SMOKE = REPO_DIR / ".smoke_deps"
if SMOKE.exists():
    sys.path.insert(0, str(SMOKE))

import scripts.realtime_d435i as rt  # noqa: E402


def _stub_calib() -> rt.CamCalib:
    return rt.CamCalib(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        width=640, height=480, baseline_m=0.05,
    )


class _FakeCamera:
    """Stand-in for :class:`rt.D435i` that yields synthetic IR pairs."""

    def __init__(self, args) -> None:
        self.args = args
        H, W = args.height, args.width
        rng = np.random.default_rng(0)
        pattern = ((np.add.outer(np.arange(H), np.arange(W)) // 32) % 2 * 200 + 30
                   ).astype(np.uint8)
        self._left = pattern
        self._right = np.roll(pattern, -30, axis=1)
        self._noise_rng = rng

    def start(self) -> rt.CamCalib:
        return _stub_calib()

    def read_pair(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        n_l = self._noise_rng.integers(0, 5, size=self._left.shape, dtype=np.int16)
        n_r = self._noise_rng.integers(0, 5, size=self._right.shape, dtype=np.int16)
        left = np.clip(self._left.astype(np.int16) + n_l, 0, 255).astype(np.uint8)
        right = np.clip(self._right.astype(np.int16) + n_r, 0, 255).astype(np.uint8)
        return left, right

    def stop(self) -> None:
        pass


def _stub_load_model(args) -> rt.LoadedModel:
    return rt.LoadedModel(
        model=SimpleNamespace(),  # never called - run_one_inference is stubbed
        args=SimpleNamespace(),
        h_input=int(round(args.height * args.scale)),
        w_input=int(round(args.width * args.scale)),
    )


def _stub_run_inference(loaded, ir_left, ir_right, args, device):
    if args.scale != 1.0:
        H = int(round(ir_left.shape[0] * args.scale))
        W = int(round(ir_left.shape[1] * args.scale))
    else:
        H, W = ir_left.shape[:2]
    disp = np.full((H, W), 30.0, dtype=np.float32)
    return disp, np.empty(0, dtype=np.float32), 5.0


def _stub_event_pair():
    """Replace torch.cuda.Event with a CPU-only timer compatible with the API."""

    class _E:
        def __init__(self, *a, **kw):
            self._t = 0.0

        def record(self):
            import time as _t
            self._t = _t.perf_counter()

        def elapsed_time(self, other):
            return (other._t - self._t) * 1000.0

    return _E


def _no_sync(*a, **kw):
    return None


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ffs_rt_smoke_"))
    print(f"[smoke] temp: {tmp}")

    # Monkey-patch heavy bits.
    rt.D435i = _FakeCamera
    rt.load_model = _stub_load_model
    rt.run_one_inference = _stub_run_inference

    import torch
    torch.cuda.Event = _stub_event_pair()
    torch.cuda.synchronize = _no_sync
    rt.torch.cuda.Event = torch.cuda.Event
    rt.torch.cuda.synchronize = _no_sync

    log_csv = tmp / "log.csv"
    rec_dir = tmp / "rec"

    common = [
        "--no-viz",
        "--max-frames", "5",
        "--warmup", "2",
        "--width", "640",
        "--height", "480",
        "--device", "cpu",
        "--log", str(log_csv),
        "--record-dir", str(rec_dir),
        "--record-every", "1",
    ]
    print("\n[smoke] === bench mode ===")
    rc = rt.main(common + ["--mode", "bench"])
    assert rc == 0, f"bench rc={rc}"
    assert log_csv.exists(), "bench CSV missing"
    n_lines = sum(1 for _ in open(log_csv))
    assert n_lines >= 4, f"expected header + ~3 rows, got {n_lines}"
    rec_subdirs = sorted(p for p in rec_dir.iterdir() if p.is_dir())
    assert rec_subdirs, "no record subdirs"
    sample = rec_subdirs[0]
    for f in ("ir_left.png", "ir_right.png", "disp.npy", "depth.npy"):
        assert (sample / f).exists(), f"missing {sample}/{f}"

    # AprilTag mode: detector loaded, no real tag in synthetic frames -> 'no tag'
    log_csv2 = tmp / "log_at.csv"
    rec_dir2 = tmp / "rec_at"
    common2 = [
        "--no-viz",
        "--max-frames", "5",
        "--warmup", "2",
        "--width", "640",
        "--height", "480",
        "--device", "cpu",
        "--tag-size", "0.080",
        "--log", str(log_csv2),
        "--record-dir", str(rec_dir2),
        "--record-every", "1",
    ]
    print("\n[smoke] === apriltag mode ===")
    rc = rt.main(common2 + ["--mode", "apriltag"])
    assert rc == 0, f"apriltag rc={rc}"
    assert log_csv2.exists(), "apriltag CSV missing"
    header = open(log_csv2).readline().strip().split(",")
    for col in ("frame_idx", "infer_ms", "fps_p50", "tag_id",
                "gt_dist_m", "gt_z_m", "depth_pred_m",
                "abs_err_m", "rel_err"):
        assert col in header, f"apriltag CSV missing column {col!r} (header={header})"

    print("\n[smoke] OK - main loop, FPS/latency, CSV, recording all working.")
    if os.environ.get("FFS_RT_SMOKE_KEEP") != "1":
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
