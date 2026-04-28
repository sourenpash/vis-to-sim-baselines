# Depth Estimation Benchmarking Harness

Unified real-time benchmark for depth-estimation models on the Intel RealSense
D435i. Three repositories live alongside this directory:

- `Fast-FoundationStereo/` (conda env: `ffs`) - stereo matcher
- `s2m2/`                  (conda env: `s2m2`) - stereo matcher
- `lingbot-depth/`         (conda env: `lingbot-depth`) - RGB-D refiner

These envs pin incompatible torch/xformers/python versions, so a single Python
process cannot host all three models. This harness sits outside the repos,
captures D435i streams to a portable session directory, and orchestrates each
adapter inside its own conda env.

## Layout

```
benchmark/
  benchmark/        Python package (camera, io, adapters, runners, metrics, viz)
  configs/          YAML configuration files
  scripts/          Shell entry points
  sessions/         Recorded sessions land here (.gitignored)
```

## Quick start

1. Create the dispatcher env (camera capture + evaluation):

    ```bash
    bash scripts/00_setup_envs.sh
    ```

   This creates `bench` and installs `pyrealsense2` into the existing
   model envs (`ffs`, `s2m2`, `lingbot-depth`) so each can read D435i
   streams in `--live` mode.

2. Record a session (D435i must be plugged in):

    ```bash
    bash scripts/01_capture.sh sessions/living_room_static --duration 30
    ```

3. Run all enabled adapters in replay mode:

    ```bash
    bash scripts/02_run_replay.sh sessions/living_room_static
    ```

4. Evaluate and produce the comparison table:

    ```bash
    bash scripts/04_evaluate.sh sessions/living_room_static
    ```

   Output: `sessions/living_room_static/results.md`,
   `results.csv`, and side-by-side PNGs under `outputs/_compare/`.

5. (Optional) Live FPS / latency for every model:

    ```bash
    bash scripts/03_run_live.sh
    ```

## Session layout

```
sessions/<name>/
  manifest.json
  ir_left/000000.png       8-bit, rectified
  ir_right/000000.png
  color/000000.png         8-bit BGR (cv2.imwrite default)
  depth/000000.png         16-bit, mm, aligned to color
  timestamps.csv
  outputs/<adapter>/
    depth/000000.npy       float32 meters, HxW
    disp/000000.npy        optional, stereo only
    conf/000000.npy        optional, s2m2 only
    timings.csv
    meta.json
```

## Adding a model

1. Create `benchmark/adapters/<name>.py` subclassing `DepthAdapter`.
2. Register a route in `benchmark/runners/run_model.py` so
   `--adapter <name>` instantiates it.
3. Add an entry to `configs/default.yaml` under `models:`.

The adapter contract:

```python
class DepthAdapter:
    name: str
    target_frame: str   # "ir_left" or "color"
    def load(self, cfg, device): ...
    def warmup(self, frame, n): ...
    def infer(self, frame) -> dict:
        # frame: ir_left, ir_right, color, raw_depth_m, intrinsics, baseline_m, manifest
        # return: {"depth_m": HxW float32, "disparity_px"?, "confidence"?}
```

## Design notes

- IR projector is captured **OFF** by default. The dot pattern is
  out-of-distribution for learned stereo matchers (FFS / S2M2). Toggle via
  `capture.emitter_enabled` in `configs/default.yaml`.
- `lingbot-depth` is a refiner: it requires an input depth map and is
  evaluated as a refiner of D435i hardware depth. It cannot run RGB-only.
- `realsense_hw` is included as a free non-learned baseline.
