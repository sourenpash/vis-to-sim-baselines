# WBCMimic Documentation

WBCMimic is an enhanced version of Mimicgen that enables more smooth data generated and controlling the chassis, lift  and arms simultaneously. Our data collection framework is built on NVIDIA Isaac Lab.

## 📚 Documentation Structure

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **Getting Started**

    ---

    Installation, setup, and your first steps with UniMimic

    [:octicons-arrow-right-24: Installation](install.md)
    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-cube-outline:{ .lg .middle } **Core Components**

    ---

    Deep dive into the architecture and WBC extensions

    [:octicons-arrow-right-24: Extensions](core-extensions.md)
    [:octicons-arrow-right-24: WBC Control](wbc-control.md)

-   :material-database:{ .lg .middle } **Data Collection**

    ---

    Generate demonstration data for robot learning

    [:octicons-arrow-right-24: Data Generation](data-generation.md)
    [:octicons-arrow-right-24: Teleoperation](teleoperation.md)

-   :material-robot:{ .lg .middle } **API Reference**

    ---

    Detailed API documentation

    [:octicons-arrow-right-24: Tasks](api_reference/tasks.md)
    [:octicons-arrow-right-24: Robots](api_reference/robots.md)

</div>

## 🚦 Quick Start

### 1. Record Demonstrations with VR
> **Prerequisites:** Ensure [Oculus Reader](https://github.com/rail-berkeley/oculus_reader) is properly configured and ready.

> Use ee-space control during collecting data.

```bash
python scripts/basic/record_demos_ur5_quest.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-OneCamera-Mimic-v0 \
    --dataset_file ./datasets/demo.hdf5 \
    --num_demos 10
```

### 2. Annotate Subtasks

> During annotation, we need the env class to be self-defined for WbcMimic, but the actions remain ee-space control. 

```bash
python scripts/mimicgen/annotate_demos.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-Mimic-Annotate-v0 \
    --input_file ./datasets/demo.hdf5 \
    --output_file ./datasets/demo_annotated.hdf5 \
    --auto
```

### 3. Generate Large-Scale Data

> During generation, use WbcMimic for better generation.

```bash
python scripts/mimicgen/generate_dataset_parallel_all.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-OneCamera-Mimic-MP-v0 \
    --input_file ./datasets/demo_annotated.hdf5 \
    --output_file ./datasets/generated.zarr \
    --generation_num_trials 100 \
    --num_envs 4 \
    --n_procs 8 \
    --distributed \
    --enable_cameras \
    --mimic_mode uni \
    --seed 42 \
    --headless
```

## 📊 Supported Tasks

| Task | Robot | Description | Subtasks |
|------|-------|-------------|----------|
| Put Bowl in Microwave & Close | UR5 | Pick bowl, place in microwave, close door | 3 |
| Clean Plate | UR5 | Fork pickup, cleaning, plate manipulation | 5 |
| Pick Toothpaste Into Cup & Push | ARX-X7 | Dual-arm mobile manipulation | 3 |

