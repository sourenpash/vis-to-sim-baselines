"""Real-time Fast-FoundationStereo on Intel RealSense D435i.

Two modes:

* ``bench``    - live IR-stereo -> depth viewer with FPS / latency HUD.
* ``apriltag`` - same depth pipeline + tag36h11 detection on the IR_left
                 frame. The AprilTag's PnP pose is the ground-truth
                 camera-to-tag distance; we compare it to the FFS depth
                 sampled at the tag center and log per-frame error.

This script lives in the FFS repo so it has direct access to ``Utils``
and ``core/`` without depending on the outer ``benchmark/`` package.

Run::

    conda activate ffs
    pip install pyrealsense2 pupil-apriltags   # one-time

    # bench mode
    python scripts/realtime_d435i.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth

    # apriltag mode (measure your printed tag with a ruler!)
    python scripts/realtime_d435i.py --mode apriltag --tag-size 0.080 \
        --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
        --log results/apriltag_run.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
sys.path.append(str(REPO_DIR))

# Local fallback for smoke-tests when pyrealsense2/pupil-apriltags are
# not installed system-wide. Harmless when the dir doesn't exist.
_SMOKE = REPO_DIR / ".smoke_deps"
if _SMOKE.exists():
    sys.path.insert(0, str(_SMOKE))

from core.utils.utils import InputPadder  # noqa: E402
from Utils import AMP_DTYPE, set_logging_format, vis_disparity  # noqa: E402


def _require(modname: str, install_hint: str):
    try:
        return __import__(modname)
    except ImportError as e:
        raise SystemExit(
            f"\n[realtime_d435i] missing dependency '{modname}'.\n"
            f"  Install in your `ffs` env: {install_hint}\n"
        ) from e


# Imported lazily because the smoke_test path may stub the camera.
def _import_realsense():
    return _require("pyrealsense2", "pip install pyrealsense2")


def _import_apriltags():
    pa = _require("pupil_apriltags", "pip install pupil-apriltags")
    return pa


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time Fast-FoundationStereo + D435i. Bench / AprilTag modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["bench", "apriltag"], default="bench",
        help="bench: FPS/latency only. apriltag: also detect a tag36h11 marker "
             "and compute distance error.",
    )
    p.add_argument(
        "--model_dir", type=str,
        default=str(REPO_DIR / "weights/23-36-37/model_best_bp2_serialize.pth"),
        help="Path to the .pth checkpoint (same as run_demo.py).",
    )
    p.add_argument("--valid_iters", type=int, default=8)
    p.add_argument("--max_disp", type=int, default=192)
    p.add_argument(
        "--optimize_build_volume", choices=["pytorch1", "triton"],
        default="pytorch1",
    )

    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--emitter", choices=["on", "off"], default="off",
        help="IR projector. 'off' is in-distribution for FFS.",
    )
    p.add_argument(
        "--scale", type=float, default=1.0,
        help="Downscale stereo input before inference (improves FPS).",
    )

    p.add_argument(
        "--tag-size", type=float, default=0.080,
        help="Side length of the tag36h11 BLACK BORDER in meters. "
             "Measure your printed tag with a ruler. The rgov/apriltag-pdfs "
             "default at scale=1 is ~0.080 m.",
    )
    p.add_argument(
        "--tag-id", type=int, default=None,
        help="If set, only this tag36h11 ID is used as ground truth "
             "(others in view are still drawn but ignored for metrics). "
             "Default: use whichever tag is detected first each frame.",
    )

    p.add_argument(
        "--log", type=str, default=None,
        help="Per-frame CSV path (apriltag mode adds tag/error columns).",
    )
    p.add_argument(
        "--record-dir", type=str, default=None,
        help="If set, dump (ir_left.png, ir_right.png, disp.npy, depth.npy, "
             "overlay.png) every --record-every frames.",
    )
    p.add_argument("--record-every", type=int, default=30)
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Auto-quit after N inference frames (warmup excluded).",
    )

    p.add_argument(
        "--no-viz", action="store_true",
        help="Disable cv2.imshow (headless / pure benchmarking).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--warmup", type=int, default=15,
        help="Warm-up iterations (excluded from FPS stats).",
    )
    return p


# ---------------------------------------------------------------------------
# Model load (mirrors scripts/run_demo.py)
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    model: "torch.nn.Module"
    args: OmegaConf
    h_input: int
    w_input: int


def load_model(args: argparse.Namespace) -> LoadedModel:
    cfg_path = Path(args.model_dir).parent / "cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg.yaml next to checkpoint: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    fused = OmegaConf.create(cfg)

    logging.info(f"Loading model from {args.model_dir}")
    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    if hasattr(model, "args"):
        model.args.valid_iters = args.valid_iters
        model.args.max_disp = args.max_disp
    device = torch.device(args.device)
    model = model.to(device).eval()

    h_in = int(round(args.height * args.scale)) if args.scale != 1.0 else args.height
    w_in = int(round(args.width * args.scale)) if args.scale != 1.0 else args.width

    logging.info(f"Warming up on dummy {h_in}x{w_in} stereo pair ...")
    dummy_l = torch.zeros((1, 3, h_in, w_in), dtype=torch.float32, device=device)
    dummy_r = torch.zeros_like(dummy_l)
    padder = InputPadder(dummy_l.shape, divis_by=32, force_square=False)
    dummy_l, dummy_r = padder.pad(dummy_l, dummy_r)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(
            dummy_l, dummy_r,
            iters=args.valid_iters, test_mode=True,
            optimize_build_volume=args.optimize_build_volume,
        )
    torch.cuda.synchronize()
    logging.info("Warm-up done.")

    return LoadedModel(model=model, args=fused, h_input=h_in, w_input=w_in)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass
class CamCalib:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline_m: float


class D435i:
    """Minimal RealSense D435i wrapper for IR-stereo capture.

    Only the IR pair is enabled (no color/depth) - we generate depth
    ourselves via FFS. Projector toggling and intrinsics + IR baseline
    extraction happen at start time.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._rs = _import_realsense()
        self._pipeline = self._rs.pipeline()
        self._calib: Optional[CamCalib] = None

    def start(self) -> CamCalib:
        rs = self._rs
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, self.args.width, self.args.height,
                          rs.format.y8, self.args.fps)
        cfg.enable_stream(rs.stream.infrared, 2, self.args.width, self.args.height,
                          rs.format.y8, self.args.fps)
        profile = self._pipeline.start(cfg)

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(
                rs.option.emitter_enabled,
                1.0 if self.args.emitter == "on" else 0.0,
            )
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0)

        ir_l = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_r = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        intr = ir_l.get_intrinsics()
        extr = ir_l.get_extrinsics_to(ir_r)
        baseline = float(np.linalg.norm(np.asarray(extr.translation, dtype=np.float64)))
        self._calib = CamCalib(
            fx=float(intr.fx), fy=float(intr.fy),
            cx=float(intr.ppx), cy=float(intr.ppy),
            width=int(intr.width), height=int(intr.height),
            baseline_m=baseline,
        )

        logging.info(
            f"D435i started: IR {self._calib.width}x{self._calib.height}@{self.args.fps} "
            f"projector={self.args.emitter} fx={self._calib.fx:.2f} "
            f"baseline={self._calib.baseline_m*1000:.2f} mm"
        )
        for _ in range(30):
            self._pipeline.wait_for_frames()
        return self._calib

    def read_pair(self, timeout_ms: int = 2000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rs = self._rs
        frames = self._pipeline.wait_for_frames(timeout_ms)
        ir_l = frames.get_infrared_frame(1)
        ir_r = frames.get_infrared_frame(2)
        if not ir_l or not ir_r:
            return None
        return (
            np.asanyarray(ir_l.get_data()).copy(),
            np.asanyarray(ir_r.get_data()).copy(),
        )

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _ir_to_tensor(ir: np.ndarray, scale: float, device: torch.device) -> torch.Tensor:
    if scale != 1.0:
        ir = cv2.resize(ir, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if ir.ndim == 2:
        ir3 = np.repeat(ir[..., None], 3, axis=-1)
    else:
        ir3 = ir[..., :3]
    t = torch.from_numpy(ir3).to(device).float()
    return t.permute(2, 0, 1).unsqueeze(0)


def run_one_inference(
    loaded: LoadedModel,
    ir_left: np.ndarray,
    ir_right: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (disparity_px, depth_m at scaled res, infer_ms)."""
    left_t = _ir_to_tensor(ir_left, args.scale, device)
    right_t = _ir_to_tensor(ir_right, args.scale, device)

    H, W = left_t.shape[-2:]
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = loaded.model.forward(
            left_t, right_t,
            iters=args.valid_iters, test_mode=True,
            optimize_build_volume=args.optimize_build_volume,
        )
    end.record()
    torch.cuda.synchronize()
    infer_ms = start.elapsed_time(end)

    disp = padder.unpad(disp.float())
    disp_np = disp.detach().cpu().numpy().reshape(H, W).clip(0, None).astype(np.float32)
    return disp_np, np.empty(0, dtype=np.float32), infer_ms


def disparity_to_depth(disp_np: np.ndarray, calib: CamCalib, scale: float) -> np.ndarray:
    fx_scaled = calib.fx * scale
    eps = 1e-6
    depth = (fx_scaled * calib.baseline_m) / np.maximum(disp_np, eps)
    depth[disp_np <= eps] = 0.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return depth


# ---------------------------------------------------------------------------
# AprilTag mode
# ---------------------------------------------------------------------------


@dataclass
class TagObservation:
    tag_id: int
    center_xy_orig: Tuple[float, float]   # in original IR_left pixel coords
    pose_R: np.ndarray                    # 3x3, tag -> camera rotation
    pose_t_m: np.ndarray                  # (3,) translation in meters
    corners_orig: np.ndarray              # 4x2 floats, original IR_left coords


def detect_tag(
    detector,
    ir_left_orig: np.ndarray,
    calib: CamCalib,
    tag_size_m: float,
) -> List[TagObservation]:
    detections = detector.detect(
        ir_left_orig,
        estimate_tag_pose=True,
        camera_params=(calib.fx, calib.fy, calib.cx, calib.cy),
        tag_size=tag_size_m,
    )
    out: List[TagObservation] = []
    for d in detections:
        if d.pose_t is None or d.pose_R is None:
            continue
        out.append(TagObservation(
            tag_id=int(d.tag_id),
            center_xy_orig=(float(d.center[0]), float(d.center[1])),
            pose_R=np.asarray(d.pose_R, dtype=np.float64).reshape(3, 3),
            pose_t_m=np.asarray(d.pose_t, dtype=np.float64).reshape(3),
            corners_orig=np.asarray(d.corners, dtype=np.float64).reshape(4, 2),
        ))
    return out


def reprojection_error_px(
    obs: TagObservation, calib: CamCalib, tag_size_m: float,
) -> float:
    """Mean per-corner reprojection error in pixels.

    Projects the 4 tag corners from tag-local frame (half-side = tag_size/2,
    z=0) through the estimated (R, t) pose with the camera intrinsics, and
    returns the average L2 distance to the detected pixel corners.

    The pose returned by pupil_apriltags (X_cam = R @ X_tag + t) is
    unambiguous, but the ordering of detection.corners[0..3] vs the model
    corners (-s,-s) (s,-s) (s,s) (-s,s) is *not* consistent across
    upstream apriltag forks (the y-axis sign of the tag-local frame
    differs between apriltag2 / apriltag3 / pupil_apriltags). Rather than
    pick one and pray, we evaluate both candidate orderings (mirrored on
    the y-axis of the tag plane) and take whichever matches the detected
    corners. Since the pose is unique, exactly one ordering can produce a
    near-zero residual; the other will produce a residual on the order of
    the tag's pixel size, which would otherwise masquerade as miscalibration.
    Returns NaN if any projected point is behind the camera.
    """
    s = tag_size_m / 2.0
    R = obs.pose_R
    t = obs.pose_t_m.reshape(3)
    pts_a = np.array(
        [[-s, -s, 0.0], [+s, -s, 0.0], [+s, +s, 0.0], [-s, +s, 0.0]],
        dtype=np.float64,
    )
    pts_b = pts_a[::-1].copy()  # y-flipped (= reversed) sibling ordering

    best = float("inf")
    for pts_tag in (pts_a, pts_b):
        pts_cam = pts_tag @ R.T + t
        z = pts_cam[:, 2]
        if not np.all(z > 1e-6):
            continue
        u = calib.fx * pts_cam[:, 0] / z + calib.cx
        v = calib.fy * pts_cam[:, 1] / z + calib.cy
        proj = np.stack([u, v], axis=1)
        err = float(np.linalg.norm(proj - obs.corners_orig, axis=1).mean())
        if err < best:
            best = err
    return best if best != float("inf") else float("nan")


def sample_depth_at(
    depth_scaled: np.ndarray,
    cx_orig: float, cy_orig: float,
    scale: float,
    half: int = 2,
) -> Optional[float]:
    """Median depth in a (2*half+1)x(2*half+1) window. Returns None if invalid."""
    H, W = depth_scaled.shape
    cx = int(round(cx_orig * scale))
    cy = int(round(cy_orig * scale))
    x0, x1 = max(cx - half, 0), min(cx + half + 1, W)
    y0, y1 = max(cy - half, 0), min(cy + half + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_scaled[y0:y1, x0:x1]
    valid = patch[(patch > 0) & np.isfinite(patch)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


# ---------------------------------------------------------------------------
# HUD / viz
# ---------------------------------------------------------------------------


def colorize_depth(depth_m: np.ndarray, vmin: float = 0.2, vmax: float = 5.0) -> np.ndarray:
    valid = (depth_m > 0) & np.isfinite(depth_m)
    d = depth_m.copy()
    d[~valid] = vmin
    norm = np.clip((d - vmin) / max(vmax - vmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[~valid] = (0, 0, 0)
    return out


def put_hud(img: np.ndarray, lines: List[str]) -> np.ndarray:
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h_text = 22 * len(lines) + 8
    cv2.rectangle(out, (0, 0), (out.shape[1], h_text), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_tag_overlay(
    bgr: np.ndarray,
    obs: TagObservation,
    gt_z_m: float,
    pred_z_m: Optional[float],
) -> np.ndarray:
    out = bgr.copy()
    pts = obs.corners_orig.astype(np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True,
                  color=(0, 255, 255), thickness=2)
    cx, cy = int(round(obs.center_xy_orig[0])), int(round(obs.center_xy_orig[1]))
    cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)
    label = f"id={obs.tag_id} gt={gt_z_m:.3f}m"
    if pred_z_m is not None:
        err_mm = (pred_z_m - gt_z_m) * 1000.0
        label += f" pred={pred_z_m:.3f}m err={err_mm:+.1f}mm"
    cv2.putText(out, label, (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# CSV / record helpers
# ---------------------------------------------------------------------------


class CSVLogger:
    def __init__(self, path: Path, header: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(header)
        self._f.flush()

    def write(self, row: List) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def save_record(
    record_dir: Path,
    idx: int,
    ir_left: np.ndarray,
    ir_right: np.ndarray,
    disp: np.ndarray,
    depth: np.ndarray,
    overlay: Optional[np.ndarray],
) -> None:
    sub = record_dir / f"{idx:06d}"
    sub.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sub / "ir_left.png"), ir_left)
    cv2.imwrite(str(sub / "ir_right.png"), ir_right)
    np.save(sub / "disp.npy", disp)
    np.save(sub / "depth.npy", depth)
    if overlay is not None:
        cv2.imwrite(str(sub / "overlay.png"), overlay)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


_QUIT_REQUESTED = False


def _install_sigint_handler() -> None:
    def _handler(signum, frame):
        global _QUIT_REQUESTED
        _QUIT_REQUESTED = True
        logging.info("\n[realtime_d435i] quit requested (SIGINT) ...")
    signal.signal(signal.SIGINT, _handler)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    set_logging_format()
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    _install_sigint_handler()

    loaded = load_model(args)
    cam = D435i(args)
    calib = cam.start()

    detector = None
    if args.mode == "apriltag":
        pa = _import_apriltags()
        detector = pa.Detector(
            families="tag36h11", nthreads=2,
            quad_decimate=1.0, refine_edges=True,
        )
        if args.tag_id is None:
            logging.info(
                f"AprilTag detector: tag36h11, tag_size={args.tag_size:.4f} m, "
                f"id_filter=any (using first tag detected each frame)"
            )
        else:
            logging.info(
                f"AprilTag detector: tag36h11, tag_size={args.tag_size:.4f} m, "
                f"id_filter={args.tag_id}"
            )

    logger: Optional[CSVLogger] = None
    if args.log:
        if args.mode == "apriltag":
            header = [
                "frame_idx", "host_t_s", "infer_ms", "fps_p50",
                "tag_id", "gt_dist_m", "gt_z_m", "depth_pred_m",
                "abs_err_m", "rel_err", "reproj_px",
            ]
        else:
            header = ["frame_idx", "host_t_s", "infer_ms", "fps_p50"]
        logger = CSVLogger(Path(args.log), header)

    record_dir = Path(args.record_dir) if args.record_dir else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    ring_total = collections.deque(maxlen=60)
    ring_infer = collections.deque(maxlen=60)
    n_inferred = 0
    abs_errs: List[float] = []
    rel_errs: List[float] = []
    reproj_errs: List[float] = []
    n_frames_with_tag = 0
    seen_tag_ids: set = set()

    t_start = time.perf_counter()

    try:
        idx = 0
        while not _QUIT_REQUESTED:
            if args.max_frames is not None and n_inferred >= args.max_frames:
                break
            pair = cam.read_pair()
            if pair is None:
                continue
            ir_l, ir_r = pair

            t_total0 = time.perf_counter()
            disp, _depth_unused, infer_ms = run_one_inference(
                loaded, ir_l, ir_r, args, device,
            )
            depth_scaled = disparity_to_depth(disp, calib, args.scale)
            total_ms = (time.perf_counter() - t_total0) * 1000.0

            is_warmup = idx < args.warmup
            if not is_warmup:
                ring_total.append(total_ms)
                ring_infer.append(infer_ms)
                n_inferred += 1

            fps_p50 = (1000.0 / np.median(ring_total)) if ring_total else 0.0
            infer_p50 = float(np.median(ring_infer)) if ring_infer else infer_ms

            obs_list: List[TagObservation] = []
            if detector is not None:
                obs_list = detect_tag(detector, ir_l, calib, args.tag_size)

            for o in obs_list:
                if o.tag_id not in seen_tag_ids:
                    seen_tag_ids.add(o.tag_id)
                    logging.info(f"  saw tag36h11 id={o.tag_id} for the first time")

            if args.tag_id is not None:
                obs_list = [o for o in obs_list if o.tag_id == args.tag_id]
            primary_obs: Optional[TagObservation] = obs_list[0] if obs_list else None
            pred_z: Optional[float] = None
            gt_z: Optional[float] = None
            reproj_px: Optional[float] = None
            if primary_obs is not None:
                gt_z = float(primary_obs.pose_t_m[2])
                pred_z = sample_depth_at(
                    depth_scaled,
                    primary_obs.center_xy_orig[0], primary_obs.center_xy_orig[1],
                    args.scale,
                )
                rep = reprojection_error_px(primary_obs, calib, args.tag_size)
                if np.isfinite(rep):
                    reproj_px = float(rep)
                    if not is_warmup:
                        reproj_errs.append(reproj_px)
                if pred_z is not None and gt_z > 0 and not is_warmup:
                    abs_err = pred_z - gt_z
                    rel_err = abs_err / gt_z
                    abs_errs.append(abs_err)
                    rel_errs.append(rel_err)
                    n_frames_with_tag += 1

            if logger is not None and not is_warmup:
                row_t = time.perf_counter() - t_start
                if args.mode == "apriltag":
                    if primary_obs is not None and gt_z is not None:
                        gt_dist = float(np.linalg.norm(primary_obs.pose_t_m))
                        abs_err = (pred_z - gt_z) if pred_z is not None else float("nan")
                        rel_err = (abs_err / gt_z) if pred_z is not None else float("nan")
                        logger.write([
                            idx, f"{row_t:.4f}", f"{infer_ms:.3f}", f"{fps_p50:.2f}",
                            primary_obs.tag_id, f"{gt_dist:.4f}", f"{gt_z:.4f}",
                            f"{pred_z:.4f}" if pred_z is not None else "",
                            f"{abs_err:.4f}" if pred_z is not None else "",
                            f"{rel_err:.4f}" if pred_z is not None else "",
                            f"{reproj_px:.3f}" if reproj_px is not None else "",
                        ])
                    else:
                        logger.write([
                            idx, f"{row_t:.4f}", f"{infer_ms:.3f}", f"{fps_p50:.2f}",
                            "", "", "", "", "", "", "",
                        ])
                else:
                    logger.write([idx, f"{row_t:.4f}", f"{infer_ms:.3f}", f"{fps_p50:.2f}"])

            depth_vis_scaled = colorize_depth(depth_scaled)
            if args.scale != 1.0:
                depth_vis = cv2.resize(
                    depth_vis_scaled, (calib.width, calib.height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                depth_vis = depth_vis_scaled

            hud_lines = [
                f"[{args.mode}] FPS {fps_p50:5.1f}  infer p50 {infer_p50:5.1f} ms  "
                f"infer now {infer_ms:5.1f} ms",
            ]
            if args.mode == "apriltag":
                if primary_obs is not None:
                    if pred_z is not None and gt_z is not None:
                        err_mm = (pred_z - gt_z) * 1000.0
                        hud_lines.append(
                            f"tag {primary_obs.tag_id}: gt {gt_z:.3f} m  "
                            f"pred {pred_z:.3f} m  err {err_mm:+.1f} mm"
                        )
                    else:
                        hud_lines.append(
                            f"tag {primary_obs.tag_id}: gt unknown / depth invalid"
                        )
                    rep_str = f"{reproj_px:.2f} px" if reproj_px is not None else "n/a"
                    hud_lines.append(f"reproj {rep_str}  (intrinsics+tag-size sanity)")
                    # If reprojection error is huge AND pred disagrees with PnP
                    # gt_z by a clean ratio, the user's --tag-size is almost
                    # certainly wrong. Suggest the corrected value.
                    if (reproj_px is not None and reproj_px > 5.0
                            and pred_z is not None and gt_z is not None
                            and pred_z > 0.05 and gt_z > 0.05):
                        ratio = pred_z / gt_z
                        if 0.4 < ratio < 0.9 or 1.1 < ratio < 2.5:
                            suggested = args.tag_size * ratio
                            hud_lines.append(
                                f"warn: try --tag-size {suggested*100:.1f} cm "
                                f"(currently {args.tag_size*100:.1f} cm)"
                            )
                else:
                    hud_lines.append("no tag")

            ir_vis = put_hud(ir_l, hud_lines)
            depth_hud = put_hud(depth_vis, hud_lines)

            overlay_for_record = depth_hud
            if primary_obs is not None and args.mode == "apriltag":
                ir_vis = draw_tag_overlay(ir_vis, primary_obs, gt_z or 0.0, pred_z)
                depth_hud = draw_tag_overlay(depth_hud, primary_obs, gt_z or 0.0, pred_z)
                overlay_for_record = depth_hud

            if not args.no_viz:
                cv2.imshow("ir_left", ir_vis)
                cv2.imshow("depth", depth_hud)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):  # q or Esc
                    break

            if record_dir is not None and (idx % args.record_every == 0):
                save_record(record_dir, idx, ir_l, ir_r, disp, depth_scaled, overlay_for_record)

            if not is_warmup and (n_inferred % 30 == 0):
                logging.info(
                    f"  frame {idx:5d}  fps {fps_p50:5.1f}  "
                    f"infer {infer_ms:5.1f} ms (p50 {infer_p50:5.1f})"
                )

            idx += 1
    finally:
        cam.stop()
        if logger is not None:
            logger.close()
        if not args.no_viz:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print()
    print(f"=== {args.mode.upper()} run summary ===")
    print(f"frames inferred  : {n_inferred} ({elapsed:.1f}s wall, "
          f"{n_inferred/max(elapsed,1e-6):.2f} avg fps)")
    if ring_total:
        print(f"infer p50        : {float(np.median(ring_infer)):.2f} ms")
        print(f"infer p90        : {float(np.percentile(ring_infer, 90)):.2f} ms")
        print(f"fps p50          : {1000.0/np.median(ring_total):.2f}")
    if args.mode == "apriltag":
        if seen_tag_ids:
            ids_str = ", ".join(str(i) for i in sorted(seen_tag_ids))
            filter_str = (f" (filter --tag-id {args.tag_id})"
                          if args.tag_id is not None else "")
            print(f"tag ids observed : {ids_str}{filter_str}")
        print(f"frames with tag  : {n_frames_with_tag} / {n_inferred}"
              f" ({100.0*n_frames_with_tag/max(n_inferred,1):.1f}%)")
        if abs_errs:
            arr_abs = np.asarray(abs_errs) * 1000.0
            arr_rel = np.asarray(rel_errs)
            print(f"mean abs err     : {arr_abs.mean():+.2f} mm")
            print(f"median abs err   : {float(np.median(np.abs(arr_abs))):.2f} mm "
                  f"(unsigned)")
            print(f"p90  abs err     : {float(np.percentile(np.abs(arr_abs), 90)):.2f} mm "
                  f"(unsigned)")
            print(f"mean rel err     : {arr_rel.mean()*100:+.2f}%")
        else:
            print("no valid tag observations were collected")
        if reproj_errs:
            arr_rep = np.asarray(reproj_errs)
            mean_rep = float(arr_rep.mean())
            med_rep = float(np.median(arr_rep))
            p90_rep = float(np.percentile(arr_rep, 90))
            if mean_rep < 0.75:
                verdict = "excellent (intrinsics + tag-size are consistent)"
            elif mean_rep < 1.5:
                verdict = "good"
            elif mean_rep < 3.0:
                verdict = "marginal -- re-measure printed tag with calipers"
            else:
                verdict = "poor -- intrinsics or tag-size likely wrong"
            print()
            print(f"reproj mean      : {mean_rep:.3f} px")
            print(f"reproj median    : {med_rep:.3f} px")
            print(f"reproj p90       : {p90_rep:.3f} px")
            print(f"reproj verdict   : {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
