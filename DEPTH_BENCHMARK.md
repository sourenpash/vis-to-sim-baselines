# Depth-Estimation Benchmark for Quadruped Locomotion — Handoff

> **Audience**: Future AI coding agents (and humans) picking up this project.
> Read this first before touching any code. It captures the **goal**, **what
> exists today**, **what is planned**, the **conventions you must respect**,
> and a few **non-obvious gotchas we already debugged** so you don't redo
> them.

---

## 1. Project goal

Build a standardized real-time benchmarking pipeline that compares multiple
depth-estimation models on **identical input frames** captured from an
**Intel RealSense D435i**, so we can pick the model that produces the most
reliable, smoothest, and lowest-latency depth for **quadruped locomotion
policies** (think near-field obstacle avoidance, 0.2–3 m range).

Comparison axes:

| Axis | Why it matters for quadruped policies |
| --- | --- |
| Per-frame inference latency (p50 / p90) | Locomotion control loops run at 30–100 Hz |
| Throughput (FPS) | Determines achievable control rate |
| Accuracy on metric distance | Foot placement, slope estimation |
| Temporal stability | Prevents jittery footstep planning |
| Spatial noise / completeness | Smooth value functions, fewer holes |
| Near-field quality (0.2–3 m) | The only range a quadruped really cares about |
| Robustness (low texture, reflective, motion) | Real-world deployment |

---

## 2. Models in scope

All three live as **separate Conda environments** because their
PyTorch / xformers / CUDA versions are mutually incompatible.

| Model | Conda env | Repo dir | Input | Output |
| --- | --- | --- | --- | --- |
| Fast-FoundationStereo (FFS) | `ffs` | `Fast-FoundationStereo/` | IR-stereo pair | Disparity → depth |
| LingBot-Depth | `lingbot-depth` | `lingbot-depth/` | RGB + raw depth | Refined depth |
| S2M2 | `s2m2` | `s2m2/` | IR-stereo pair | Disparity → depth |
| **D435i hardware depth** | _(any)_ | (passthrough) | Built-in stereo block | Depth (baseline reference) |

**Hardware**: Intel RealSense D435i. Native streams used:

- IR_left + IR_right: Y8, configurable (we use **848×480 @ 30 fps**)
- Depth: Z16, hardware-block (used by `realsense_hw` baseline + LingBot)
- Color: optional (used by LingBot only)

**IR projector default**: `--emitter off`. The projector adds an active
speckle pattern that's helpful for the D435i's hardware depth block but is
*out of distribution* for FFS / S2M2 (their training data is mostly
clean stereo IR / RGB). Default off; you can flip it on per-run.

---

## 3. What exists today

### 3.1 `Fast-FoundationStereo/scripts/realtime_d435i.py` ✅ COMPLETE

A self-contained, real-time live-camera benchmarking script for **FFS only**.
Runs entirely inside the `ffs` Conda env with no dependency on a shared
benchmark package. This is the **prototype / template** that the S2M2 and
LingBot equivalents should mirror.

#### Features

- **Two modes** via `--mode`:
  - `bench`: live IR-stereo → FFS depth, FPS/latency HUD only.
  - `apriltag`: same, plus tag36h11 ground-truth distance + reprojection-error sanity check.
- **D435i pipeline**: enables IR_left + IR_right at chosen res/fps, toggles emitter, reads factory intrinsics + IR baseline, warms up 30 frames before timing starts.
- **Inference**: `core.utils.utils.InputPadder` → `torch.amp.autocast(dtype=AMP_DTYPE)` → FFS forward → disparity → depth via `depth = fx_scaled * baseline / disparity`. Mirrors `scripts/run_demo.py` exactly.
- **Timing**: `torch.cuda.Event` for inference-only ms; `time.perf_counter` for end-to-end. Rolling 60-frame deque for p50/p90.
- **HUD**: live `cv2.imshow` of IR_left + colorized depth, with FPS, infer-now/p50, mode-specific lines (tag id, gt, pred, err, reproj).
- **CSV log** (`--log`): one row per frame, schema differs by mode (see §6).
- **Record dump** (`--record-dir`): every N frames saves `ir_left.png`, `ir_right.png`, `disp.npy`, `depth.npy`, `overlay.png` into a per-frame subdir. Used to feed the offline evaluator and to spot-check failures.
- **Clean exit**: `q` / Esc / SIGINT all flush the CSV, close the camera, and print a summary table.
- **AprilTag ground truth** (mode=`apriltag`):
  - tag36h11 detector via `pupil_apriltags`
  - `--tag-size` (meters, **outer black border**)
  - `--tag-id` optional filter; otherwise uses first detected tag each frame
  - PnP gives ground-truth `gt_z` (camera-to-tag distance)
  - We sample model depth at the tag center (5×5 median window, ignores zeros) → `pred_z`
  - Per-frame `abs_err = pred_z - gt_z`, `rel_err = abs_err/gt_z`
  - **Reprojection error sanity check** (see §5)

#### Files

- `Fast-FoundationStereo/scripts/realtime_d435i.py` — the script.
- `Fast-FoundationStereo/scripts/_smoke_realtime.py` — offline smoke test that monkey-patches camera/model/inference so it runs on a CPU-only box without a D435i. Tests argparse → frame loop → CSV → record → summary, plus the `reprojection_error_px` math.

#### Dependencies (one-time, in `ffs` env)

```bash
conda activate ffs
pip install pyrealsense2 pupil-apriltags
```

### 3.2 What's NOT built yet

The shell-script orchestrator package (`scripts/00_setup_envs.sh` etc.) and
the cross-env evaluator (`benchmark/` package) **mentioned in earlier
planning notes have not been written**. Only the FFS live script and its
smoke test are on disk. The next agent should either:

- (a) Build a per-model live script for **S2M2** and **LingBot-Depth** that
  mirrors `realtime_d435i.py`'s API (HUD/CSV/record contract), then aggregate
  results with a small comparison script. Recommended path; least new code.
- (b) Build the on-disk session contract (`SessionWriter`/`SessionReader`)
  and a thin `DepthAdapter` ABC so all three models share one runner; this
  is heavier but gives strict fairness.

Either path needs to preserve the **CSV/record schema below** so outputs are
comparable.

---

## 4. AprilTag ground-truth methodology

The point of the `apriltag` mode is to provide an **online, transparent
ground truth** for metric depth without ever asking the user to set up a
motion-capture rig or do offline manual annotation. Hold a printed tag at
arbitrary distances, the camera tells you the real distance, the model
tells you its predicted distance, the script logs the error.

### 4.1 Why we don't need to recalibrate the camera

D435i intrinsics come from Intel's factory calibration. The script reads
`fx, fy, ppx, ppy` directly from the IR_left stream profile via
`pyrealsense2`, and reads the IR-pair extrinsic translation (the stereo
baseline) from `get_extrinsics_to(...)`. These are accurate to better than
~0.5 % out of the box. Recalibration would only matter if (a) the camera
suffered a hard impact, or (b) reprojection error stays > 2 px after
fixing tag-size.

### 4.2 Tag-size convention — read this carefully

`pupil_apriltags` and the upstream AprilTag library both define
`tag_size` as:

> **The side length of the OUTER black border of the tag, measured edge to
> edge in meters. NOT including the white quiet zone around the tag.**

If you measure the **biggest black square** with a ruler, that's the value
you pass to `--tag-size`. For our printed tag this is **0.16 m**.

Common alternative conventions you might trip over:

| Library | "tag size" means |
| --- | --- |
| `pupil_apriltags` (we use this) | **Outer black border** ✅ |
| ROS `apriltag_ros` (older) | Inner black-border edge (= 8/10 × outer) |
| ArUco tutorials | Often the entire marker including white quiet zone |

If your `--tag-size` is wrong by factor *r*, then **PnP returns gt_z scaled
by 1/r**, and the live HUD will print:

```text
warn: try --tag-size XX.X cm (currently YY.Y cm)
```

We added that warning specifically because we hit it once already.

### 4.3 Reprojection error — what it tells you

After PnP gives a pose, we project the 4 tag corners back through the
factory intrinsics and compare to the corners the detector found. The
mean L2 distance is `reproj_px`.

| `reproj_px` | Meaning |
| --- | --- |
| < 0.75 px | **Excellent** — intrinsics + tag-size are mutually consistent. Trust everything. |
| 0.75–1.5 px | **Good** — ship it. |
| 1.5–3 px | **Marginal** — re-measure printed tag with calipers. |
| > 3 px | **Poor** — `--tag-size` likely wrong, or D435i was physically damaged. |

The script prints a verdict line at end-of-run (mean, median, p90,
human-readable verdict).

### 4.4 The corner-ordering bug we already fixed (DON'T REGRESS)

`pupil_apriltags` returns `detection.corners[0..3]` in image-CCW order.
Different forks of the upstream `apriltag_pose.c` use **different signs
for the y-component of the model-corner table**. Specifically the model
points are either:

```text
A: [(-s,-s), (+s,-s), (+s,+s), (-s,+s)]
B: [(-s,+s), (+s,+s), (+s,-s), (-s,-s)]   # reversed of A, equivalent y-flip
```

Both produce identical PnP poses (the optimization is invariant to a
y-flip of the tag plane), but the *back-projection step* in our reprojection
metric is **not** invariant — picking the wrong ordering produces a
residual exactly equal to **`fx · tag_size / depth`** pixels, which for our
setup (fx≈416, tag=0.16 m, depth=0.34 m) is ~195 px and looks like a
camera-calibration disaster.

**Fix**: `reprojection_error_px` evaluates **both** orderings and keeps the
smaller residual. The pose is unique, so only one ordering can possibly
align — we use whichever does. Don't simplify this to a single ordering or
you'll re-introduce the bug the next time `pupil_apriltags` updates.

---

## 5. Output / deliverable schema

Every model's runner MUST produce these for fair comparison.

### 5.1 Per-frame CSV (`--log results/<run>.csv`)

**bench mode** (4 columns):

```
frame_idx, host_t_s, infer_ms, fps_p50
```

**apriltag mode** (11 columns):

```
frame_idx, host_t_s, infer_ms, fps_p50,
tag_id, gt_dist_m, gt_z_m, depth_pred_m,
abs_err_m, rel_err, reproj_px
```

- `gt_dist_m` = `‖t‖` (Euclidean from camera to tag center)
- `gt_z_m`    = `t[2]` (depth component, what we compare against)
- `depth_pred_m` = model depth at tag center, 5×5 median window
- `abs_err_m` = `depth_pred_m − gt_z_m` (signed)
- `rel_err`   = `abs_err_m / gt_z_m`
- `reproj_px` = mean per-corner reprojection error (sub-pixel = good)

Empty cells when no tag is in view — keep the column count fixed.

### 5.2 Recorded artifacts (`--record-dir`)

```
<record-dir>/
├── 000000/
│   ├── ir_left.png
│   ├── ir_right.png
│   ├── disp.npy        # float32 disparity in pixels (model output)
│   ├── depth.npy       # float32 depth in meters
│   └── overlay.png     # HUD-annotated viz (for sanity scrolling)
├── 000030/
│   └── ...
```

Saved every `--record-every` frames. `disp.npy` and `depth.npy` are at
**inference resolution** (i.e., post-`--scale`), not native camera res.
Document `--scale` in the run name.

### 5.3 End-of-run summary (stdout)

Every run prints:

```text
=== APRILTAG run summary ===
frames inferred  : 87 (3.0s wall, 29.0 avg fps)
infer p50        : 18.42 ms
infer p90        : 21.10 ms
fps p50          : 29.50
tag ids observed : 0  (filter --tag-id 0)
frames with tag  : 85 / 87 (97.7%)
mean abs err     : +4.20 mm
median abs err   : 3.50 mm (unsigned)
p90  abs err     : 8.10 mm (unsigned)
mean rel err     : +1.23%

reproj mean      : 0.42 px
reproj median    : 0.41 px
reproj p90       : 0.59 px
reproj verdict   : excellent (intrinsics + tag-size are consistent)
```

When you build the cross-model comparison, **scrape these summary blocks**
or better, re-aggregate from CSVs. Don't trust hand-typed numbers.

---

## 6. Quick-start commands

### Live FFS bench (no ground truth, just FPS/latency)

```bash
conda activate ffs
cd Fast-FoundationStereo
python scripts/realtime_d435i.py \
    --mode bench \
    --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
    --log results/ffs_bench.csv \
    --record-dir results/ffs_bench_frames
```

### Live FFS with AprilTag ground truth (recommended)

```bash
conda activate ffs
cd Fast-FoundationStereo
python scripts/realtime_d435i.py \
    --mode apriltag --tag-size 0.16 --tag-id 0 \
    --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
    --log results/ffs_apriltag.csv \
    --record-dir results/ffs_apriltag_frames
```

### Headless (no GUI, useful over SSH or in CI)

Add `--no-viz --max-frames 300` to either of the above.

### Smoke test (no camera / no GPU)

```bash
cd Fast-FoundationStereo
PYTHONPATH=.smoke_deps python scripts/_smoke_realtime.py
```

---

## 7. Conventions / gotchas for future agents

1. **The `ffs`/`s2m2`/`lingbot-depth` Conda envs have incompatible deps.**
   Don't try to make a single env. Each model runs in its own env. The only
   thing they share is the **on-disk output format** (§5).

2. **D435i factory intrinsics are good enough.** Don't add a calibration
   step. If `reproj_px` is high, the problem is `--tag-size`, not the
   camera. We already shipped a HUD warning for that.

3. **`--tag-size` = OUTER BLACK BORDER**, measured with a ruler / calipers.
   Not the inner border, not the data-bit area, not the marker including
   the white quiet zone. The corresponding HUD warning will suggest the
   right value if you get this wrong.

4. **Don't single-order the reprojection back-projection.** The
   `reprojection_error_px` helper tries both y-sign conventions because
   pupil_apriltags' internal `apriltag_pose.c` y-sign convention is not
   stable across forks. We hit a 195-px false-positive doing this naively;
   it's documented in §4.4.

5. **Inference resolution vs camera resolution.** When `--scale != 1.0`,
   `disp.npy` and `depth.npy` are at the **scaled** resolution. Use
   `calib.fx * scale` for the disparity-to-depth conversion (we do).
   The `depth_pred_m` sample location must scale the tag pixel coords by
   `--scale` before indexing the depth map (we do).

6. **Tag ID is logged but not auto-filtered by default.** If you have
   multiple tags in view, pin one with `--tag-id N`. The HUD prints
   `saw tag36h11 id=X for the first time` so you know what's in view.

7. **The IR projector should be OFF for FFS / S2M2** (out-of-distribution
   for their training data). It can be ON for the D435i hardware-depth
   baseline if you want to characterise it at its best. Make this explicit
   in the run's filename / README.

8. **Per-model CSV schema must match exactly.** If you add a column,
   add it to all three models' runners or you'll break the comparison
   evaluator. Better: add new metrics as a *new file* (`extra_metrics.csv`)
   so the canonical schema is stable.

9. **Don't commit weights**. The repo's `.gitignore` (root and per-model)
   already excludes `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`, `*.onnx`,
   `pretrain_weights/`, `weights/`, `checkpoint*/`. Verify before committing.

---

## 8. Status and next steps

**Done**

- [x] D435i live-camera FFS pipeline with FPS/latency HUD
- [x] AprilTag tag36h11 ground-truth integration (PnP-based)
- [x] Reprojection-error sanity metric with verdict thresholds
- [x] Tag-size suggestion warning when reproj is large
- [x] Tag-ID filtering and per-ID first-seen logging
- [x] CSV + record-dir + summary-print contract (§5)
- [x] Offline smoke test (no camera, no GPU)
- [x] Bug fix: ordering-robust reprojection metric (§4.4)

**Next**

- [ ] **S2M2** equivalent of `realtime_d435i.py` in `s2m2/scripts/`
- [ ] **LingBot-Depth** equivalent in `lingbot-depth/` (note: takes RGB +
      raw D435i depth, not stereo IR — the camera setup differs)
- [ ] **D435i hardware-depth baseline** — trivial passthrough runner
      using the same CSV/record schema
- [ ] **Aggregator script** that reads `results/*.csv` from all three
      models for the same scene and produces a comparison table + plots
      (FPS vs accuracy vs near-field error)

**Stretch**

- [ ] Replay mode: record one D435i session once, replay it across all
      three models offline (perfect frame-for-frame fairness; trades real-time
      pressure for reproducibility)
- [ ] Spatial / temporal / completeness metrics from the planning doc
- [ ] Reflective-surface and motion-blur stress tests
