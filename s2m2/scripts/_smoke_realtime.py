"""Offline smoke test for ``scripts/realtime_d435i.py``.

Patches the D435i camera, S2M2 model load, and S2M2 inference call so the
live runner can be tested in the ``s2m2`` Conda env without a RealSense
camera, CUDA, or a printed tag.

Run::

    conda activate s2m2
    python scripts/_smoke_realtime.py
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
sys.path.insert(0, str(REPO_DIR / "src"))

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
        pattern = (
            (np.add.outer(np.arange(H), np.arange(W)) // 32) % 2 * 200 + 30
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
        model=SimpleNamespace(),
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


def _check_reprojection_math() -> None:
    """Verify the ordering-robust reprojection helper on a perfect pose."""
    calib = _stub_calib()
    tag_size = 0.080
    s = tag_size / 2.0
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.5])
    pts_tag = np.array(
        [[-s, -s, 0.0], [+s, -s, 0.0], [+s, +s, 0.0], [-s, +s, 0.0]],
        dtype=np.float64,
    )
    pts_cam = pts_tag @ R.T + t
    u = calib.fx * pts_cam[:, 0] / pts_cam[:, 2] + calib.cx
    v = calib.fy * pts_cam[:, 1] / pts_cam[:, 2] + calib.cy
    corners = np.stack([u, v], axis=1)

    obs = rt.TagObservation(
        tag_id=0,
        center_xy_orig=(calib.cx, calib.cy),
        pose_R=R,
        pose_t_m=t,
        corners_orig=corners,
    )
    err_perfect = rt.reprojection_error_px(obs, calib, tag_size)
    assert err_perfect < 1e-6, (
        f"perfect-pose reproj err should be ~0, got {err_perfect}"
    )

    err_wrong = rt.reprojection_error_px(obs, calib, tag_size * 1.10)
    assert err_wrong > 1.0, (
        f"10%-wrong tag-size should produce >1 px error, got {err_wrong}"
    )
    print(
        f"[smoke] reproj math: perfect={err_perfect:.2e} px  "
        f"10%-wrong-size={err_wrong:.2f} px  OK"
    )


def _assert_record_sample(rec_dir: Path) -> None:
    rec_subdirs = sorted(p for p in rec_dir.iterdir() if p.is_dir())
    assert rec_subdirs, "no record subdirs"
    sample = rec_subdirs[0]
    for f in ("ir_left.png", "ir_right.png", "disp.npy", "depth.npy", "overlay.png"):
        assert (sample / f).exists(), f"missing {sample}/{f}"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="s2m2_rt_smoke_"))
    print(f"[smoke] temp: {tmp}")

    _check_reprojection_math()

    rt.D435i = _FakeCamera
    rt.load_model = _stub_load_model
    rt.run_one_inference = _stub_run_inference

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
    header = open(log_csv).readline().strip().split(",")
    assert header == ["frame_idx", "host_t_s", "infer_ms", "fps_p50"], header
    n_lines = sum(1 for _ in open(log_csv))
    assert n_lines == 6, f"expected header + 5 rows, got {n_lines}"
    _assert_record_sample(rec_dir)

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
    assert header == [
        "frame_idx", "host_t_s", "infer_ms", "fps_p50",
        "tag_id", "gt_dist_m", "gt_z_m", "depth_pred_m",
        "abs_err_m", "rel_err", "reproj_px",
    ], header
    n_lines = sum(1 for _ in open(log_csv2))
    assert n_lines == 6, f"expected header + 5 rows, got {n_lines}"
    _assert_record_sample(rec_dir2)

    print("\n[smoke] OK - main loop, FPS/latency, CSV, recording all working.")
    if os.environ.get("S2M2_RT_SMOKE_KEEP") != "1":
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
