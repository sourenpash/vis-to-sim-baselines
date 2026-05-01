# Getting Started

## Prerequisites

Before you begin, ensure you have:

- ✅ Completed the [installation](install.md) process
- ✅ NVIDIA RTX GPU with at least 8GB VRAM
- ✅ Ubuntu 20.04 or 22.04

## Quick Start: Data Collection Workflow

WBCMimic uses a three-step process for generating large-scale demonstration data:

### Step 1: Record Demonstrations

Record initial demonstrations using VR teleoperation:

> **Prerequisites:** Ensure [Oculus Reader](https://github.com/rail-berkeley/oculus_reader) is properly configured and ready.

```bash
python scripts/basic/record_demos_ur5_quest.py \
    --task Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-v0 \
    --dataset_file ./demos/clean_plate.hdf5 \
    --num_demos 5
```

**Key controls:**

- **Quest Controllers**: Control robot end-effector

- **R key**: Reset and discard the current recording

- **C key**: Continue recording current episode

### Step 2: Annotate Subtasks

Add subtask completion annotations for MimicGen:

```bash
python scripts/mimicgen/annotate_demos.py \
    --task Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-Annotate-v0 \
    --input_file ./demos/clean_plate.hdf5 \
    --output_file ./demos/clean_plate_annotated.hdf5 \
    --auto
```

### Step 3: Generate Large-Scale Data

Use MimicGen to generate thousands of demonstrations:

```bash
python scripts/mimicgen/generate_dataset_parallel_all.py \
    --task Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-MP-v0 \
    --input_file ./demos/clean_plate_annotated.hdf5 \
    --output_file ./datasets/clean_plate_10k.zarr \
    --generation_num_trials 1250 \
    --num_envs 4 \
    --n_procs 8 \
    --distributed \
    --enable_cameras \
    --mimic_mode uni \
    --headless
```

This will generate 10,000 demonstrations (1250 × 8 processes) in parallel.

## Available Tasks

Currently supported UR5 tasks:

| Task Name | Task ID | Subtasks |
|-----------|---------|----------|
| Clean Plate | `Isaac-UR5-CleanPlate-Joint-*` | 1 |
| Put Bowl in Microwave & Close | `Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-*` | 3 |

## Understanding Task Variants

Common variants:

- **Mimic-Annotate**: Task variant for annotation (used for wbc mimicgen)

- **MP**: Multiprocessing variant. The only difference is the entry point. 

## Troubleshooting

### Common Issues

**Out of Memory Error:**

- Reduce `--num_envs` parameter

- Use `--state_only` flag to skip image recording

**Slow Data Generation:**

- Increase `--n_procs` (requires more GPUs)

- Use SSD for output directory