"""Intel RealSense D435i capture wrapper.

Configures the four streams we need (IR left/right + color + depth aligned to
color), exposes a generator that yields synchronized frames, and dumps
per-stream intrinsics + ir_left->ir_right + ir_left->color extrinsics into a
:class:`benchmark.io.session.Manifest`.

Designed for python>=3.7. The only dependency beyond numpy is pyrealsense2.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as e:  # pragma: no cover - reported at runtime
    raise ImportError(
        "pyrealsense2 is required. `pip install pyrealsense2` in this env."
    ) from e

from ..io.session import (
    CaptureMeta,
    Extrinsics,
    Manifest,
    StreamIntrinsics,
)


@dataclass
class D435iConfig:
    ir_width: int = 848
    ir_height: int = 480
    ir_fps: int = 30
    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30
    depth_width: int = 848
    depth_height: int = 480
    depth_fps: int = 30
    align_depth_to: str = "color"  # "color" or "ir_left"
    emitter_enabled: bool = False  # IR projector. False = clean IR for learned matchers
    auto_exposure: bool = True


@dataclass
class CapturedFrame:
    idx: int
    ir_left: np.ndarray         # uint8 (H, W)
    ir_right: np.ndarray        # uint8 (H, W)
    color_bgr: np.ndarray       # uint8 (H, W, 3)
    depth_mm_u16: np.ndarray    # uint16 (H_color, W_color), aligned to color
    host_t_ns: int
    ir_t_ns: int
    color_t_ns: int
    depth_t_ns: int


def _intrinsics_from_rs(intr: "rs.intrinsics") -> StreamIntrinsics:
    return StreamIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
        distortion_model=str(intr.model).lower().split(".")[-1],
        distortion_coeffs=[float(c) for c in intr.coeffs],
    )


def _extrinsics_from_rs(extr: "rs.extrinsics") -> Extrinsics:
    return Extrinsics(
        rotation=[float(r) for r in extr.rotation],
        translation=[float(t) for t in extr.translation],
    )


class D435iCapture:
    """Context-managed RealSense pipeline producing :class:`CapturedFrame`."""

    def __init__(self, cfg: D435iConfig) -> None:
        self.cfg = cfg
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._profile = None
        self._intrinsics: dict = {}
        self._extrinsics: dict = {}
        self._depth_scale_m: float = 1.0 / 1000.0

    def __enter__(self) -> "D435iCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.infrared, 1,
            self.cfg.ir_width, self.cfg.ir_height, rs.format.y8, self.cfg.ir_fps,
        )
        cfg.enable_stream(
            rs.stream.infrared, 2,
            self.cfg.ir_width, self.cfg.ir_height, rs.format.y8, self.cfg.ir_fps,
        )
        cfg.enable_stream(
            rs.stream.color,
            self.cfg.color_width, self.cfg.color_height, rs.format.bgr8,
            self.cfg.color_fps,
        )
        cfg.enable_stream(
            rs.stream.depth,
            self.cfg.depth_width, self.cfg.depth_height, rs.format.z16,
            self.cfg.depth_fps,
        )

        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(cfg)

        device = self._profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(
                rs.option.emitter_enabled, 1.0 if self.cfg.emitter_enabled else 0.0
            )
        # Auto exposure on the IR/depth sensor; D435 RGB is a separate sensor.
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(
                rs.option.enable_auto_exposure, 1.0 if self.cfg.auto_exposure else 0.0
            )
        self._depth_scale_m = float(depth_sensor.get_depth_scale())

        target = (
            rs.stream.color
            if self.cfg.align_depth_to == "color"
            else rs.stream.infrared
        )
        self._align = rs.align(target)

        ir_left = self._profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_right = self._profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        color = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()

        self._intrinsics = {
            "ir_left": _intrinsics_from_rs(ir_left.get_intrinsics()),
            "ir_right": _intrinsics_from_rs(ir_right.get_intrinsics()),
            "color": _intrinsics_from_rs(color.get_intrinsics()),
            "depth": _intrinsics_from_rs(depth.get_intrinsics()),
        }
        self._extrinsics = {
            "ir_left_to_ir_right": _extrinsics_from_rs(
                ir_left.get_extrinsics_to(ir_right)
            ),
            "ir_left_to_color": _extrinsics_from_rs(
                ir_left.get_extrinsics_to(color)
            ),
            "color_to_depth": _extrinsics_from_rs(
                color.get_extrinsics_to(depth)
            ),
        }

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    @property
    def depth_scale_m(self) -> float:
        return self._depth_scale_m

    def build_manifest(
        self,
        duration_s: float,
        n_frames: int,
        notes: Optional[str] = None,
    ) -> Manifest:
        capture = CaptureMeta(
            width=self.cfg.color_width,
            height=self.cfg.color_height,
            fps=self.cfg.color_fps,
            projector_on=bool(self.cfg.emitter_enabled),
            duration_s=float(duration_s),
            n_frames=int(n_frames),
            depth_scale_to_meters=float(self._depth_scale_m),
        )
        # Drop the synthetic depth-stream intrinsic from the manifest because the
        # depth frame is aligned to color and uses the color intrinsics post-align.
        streams = {
            k: v for k, v in self._intrinsics.items() if k != "depth"
        }
        return Manifest(
            capture=capture,
            streams=streams,
            extrinsics=self._extrinsics,
            notes=notes,
        )

    def read_one(self, timeout_ms: int = 2000) -> Optional[CapturedFrame]:
        """Block for one synchronized frame; returns ``None`` on timeout."""
        if self._pipeline is None or self._align is None:
            raise RuntimeError("D435iCapture.start() must be called before read_one()")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)

        ir_l = aligned.get_infrared_frame(1)
        ir_r = aligned.get_infrared_frame(2)
        col = aligned.get_color_frame()
        dep = aligned.get_depth_frame()
        if not ir_l or not ir_r or not col or not dep:
            return None

        ir_left = np.asanyarray(ir_l.get_data()).copy()
        ir_right = np.asanyarray(ir_r.get_data()).copy()
        color_bgr = np.asanyarray(col.get_data()).copy()
        depth_u16 = np.asanyarray(dep.get_data()).copy()
        if depth_u16.dtype != np.uint16:
            depth_u16 = depth_u16.astype(np.uint16)

        return CapturedFrame(
            idx=-1,
            ir_left=ir_left,
            ir_right=ir_right,
            color_bgr=color_bgr,
            depth_mm_u16=depth_u16,
            host_t_ns=time.time_ns(),
            ir_t_ns=int(ir_l.get_timestamp() * 1_000_000),
            color_t_ns=int(col.get_timestamp() * 1_000_000),
            depth_t_ns=int(dep.get_timestamp() * 1_000_000),
        )

    def stream(
        self,
        warmup_frames: int = 30,
        max_frames: Optional[int] = None,
    ) -> Iterator[CapturedFrame]:
        """Yield synchronized frames after a warmup period.

        ``warmup_frames`` are read but not yielded so auto-exposure has settled
        before the user sees any data. Frame indices in the yielded objects
        start at 0.
        """
        for _ in range(warmup_frames):
            self.read_one()

        i = 0
        while True:
            if max_frames is not None and i >= max_frames:
                return
            f = self.read_one()
            if f is None:
                continue
            f.idx = i
            yield f
            i += 1


@contextmanager
def open_d435i(cfg: D435iConfig) -> Iterator[D435iCapture]:
    cam = D435iCapture(cfg)
    try:
        cam.start()
        yield cam
    finally:
        cam.stop()
