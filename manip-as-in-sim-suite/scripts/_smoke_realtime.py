"""Offline smoke test for ``scripts/realtime_d435i.py``.

Patches the camera, model load, and inference call so the benchmark loop can
be checked in the ``manip-as-in-sim`` Conda env without RealSense hardware or
the 5 GB CDM checkpoint.
"""

from __future__ import annotations

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

import scripts.realtime_d435i as rt  # noqa: E402


def _stub_calib() -> rt.CamCalib:
    return rt.CamCalib(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        width=640, height=480, baseline_m=0.05, depth_scale_m=0.001,
    )


class _FakeCamera:
    def __init__(self, args) -> None:
        self.args = args
        H, W = args.height, args.width
        y, x = np.indices((H, W))
        color = np.zeros((H, W, 3), dtype=np.uint8)
        color[..., 0] = (x % 255).astype(np.uint8)
        color[..., 1] = (y % 255).astype(np.uint8)
        color[..., 2] = ((x + y) % 255).astype(np.uint8)
        self._color = color
        self._depth = np.full((H, W), 1.5, dtype=np.float32)

    def start(self) -> rt.CamCalib:
        return _stub_calib()

    def read_rgbd(self, timeout_ms: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        return self._color.copy(), self._depth.copy()

    def stop(self) -> None:
        pass


def _stub_load_model(args) -> rt.LoadedModel:
    return rt.LoadedModel(
        model=SimpleNamespace(),
        h_input=int(round(args.height * args.scale)),
        w_input=int(round(args.width * args.scale)),
    )


def _stub_run_inference(loaded, color_bgr, raw_depth_m, args, device):
    if args.scale != 1.0:
        H = int(round(color_bgr.shape[0] * args.scale))
        W = int(round(color_bgr.shape[1] * args.scale))
    else:
        H, W = color_bgr.shape[:2]
    depth = np.full((H, W), 1.5, dtype=np.float32)
    raw = np.full((H, W), 1.5, dtype=np.float32)
    return depth, raw, 7.0


def _check_reprojection_math() -> None:
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
    for f in ("rgb.png", "raw_depth.png", "raw_depth.npy", "disp.npy", "depth.npy", "overlay.png"):
        assert (sample / f).exists(), f"missing {sample}/{f}"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cdm_rt_smoke_"))
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
    header = open(log_csv2).readline().strip().split(",")
    assert header == [
        "frame_idx", "host_t_s", "infer_ms", "fps_p50",
        "tag_id", "gt_dist_m", "gt_z_m", "depth_pred_m",
        "abs_err_m", "rel_err", "reproj_px",
    ], header
    n_lines = sum(1 for _ in open(log_csv2))
    assert n_lines == 6, f"expected header + 5 rows, got {n_lines}"
    _assert_record_sample(rec_dir2)

    shutil.rmtree(tmp)
    print("\n[smoke] OK - main loop, FPS/latency, CSV, recording all working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
