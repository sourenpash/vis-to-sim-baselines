# Data Generation

## Overview

The data generation pipeline consists of three steps:

1. **Teleoperation**: Record initial demonstrations using VR controllers
2. **Annotation**: Add subtask completion markers
3. **Generation**: Create thousands of variations using MimicGen

## Step 1: Recording Demonstrations

### VR Teleoperation

Record demonstrations using Quest VR controllers:

```bash
python scripts/basic/record_demos_ur5_quest.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-Mimic-v0 \
    --dataset_file ./demos/put_bowl_demo.hdf5 \
    --num_demos 10 \
    --step_hz 30
```

**Controls:**
- **R**: Reset environment
- **C**: Continue current recording

### Recording Tips

- Record 1-20 high-quality demos
- Maintain smooth, deliberate motions

## Step 2: Annotating Subtasks

Annotate demonstrations with subtask completion signals:

```bash
python scripts/mimicgen/annotate_demos.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-Mimic-Annotate-v0 \
    --input_file ./demos/put_bowl_demo.hdf5 \
    --output_file ./demos/put_bowl_demo_annotated.hdf5 \
    --auto \
    --enable_cameras
```

### Annotation Process

The annotation system:
- Replays each demonstration
- Detects subtask completions automatically

> If WBC mode is used, you need to define a seperate annotate task using the entry point of WBC but the config of the original one. It will remark the target eef pose to make the pose be in the world frame.

### Manual Annotation

For complex tasks, use manual annotation. This should be supported be original mimicgen. But we didn't test this.

## Step 3: Large-Scale Generation

### Basic Generation

Generate demonstrations using MimicGen:

```bash
python scripts/mimicgen/generate_dataset_parallel_all.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-OneCamera-Mimic-MP-v0 \
    --input_file ./demos/put_bowl_demo_annotated.hdf5 \
    --output_file ./datasets/put_bowl_10k.zarr \
    --generation_num_trials 100 \
    --num_envs 4 \
    --enable_cameras \
    --mimic_mode uni \
    --headless
```

Note: 
- `--enable_cameras` is required if tasks have cameras (most example tasks do)
- `--mimic_mode uni` supports both WBC and keypoint modes (RL in future)

### Parallel Generation

For maximum efficiency with multiple GPUs:

```bash
python scripts/mimicgen/generate_dataset_parallel_all.py \
    --task Isaac-UR5-Task-Joint-GoHome-OneCamera-Mimic-MP-v0 \
    --input_file demo_annotated.hdf5 \
    --output_file dataset.zarr \
    --generation_num_trials 1250 \
    --num_envs 4 \
    --n_procs 8 \
    --distributed \
    --enable_cameras \
    --mimic_mode uni \
    --headless \
    --seed 42
```

This generates 10,000 demos (1250 × 8) across 8 GPUs.

### Hydra Configuration

For flexible configuration management, use Hydra to launch data generation:

```bash
# Use default configuration
python scripts/mimicgen/run_hydra.py

# Use specific task configuration
python scripts/mimicgen/run_hydra.py --config-name config/user/generation/generate_ur5_clean_plate_75_high_arm_mimicgen

# Override specific parameters
python scripts/mimicgen/run_hydra.py \
    task=Isaac-UR5-CleanPlate-Joint-Mimic-v0 \
    generation_num_trials=500 \
    num_envs=8
```

Configuration files are located in `config/` directory. See the [Configuration](#configuration) section for details.

### Generation Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--generation_num_trials` | Demos per process | 100-2000 |
| `--num_envs` | Parallel environments per GPU | 4-16 |
| `--n_procs` | Number of GPUs/processes | 1-8 |
| `--mimic_mode` | Control mode (origin/wbc/uni) | uni |
| `--state_only` | Skip image recording | For testing or repalying|
| `--record_all` | Keep failed demos | False |

## Data Formats

### Zarr (Recommended)

Efficient storage with compression:

```python
# Structure of generated Zarr dataset
dataset.zarr/
├── data/
│   ├── keys
│   ├── ...
└── meta  # Metadata
```

### HDF5

Compatible with robomimic:

```python
# HDF5 structure
demos.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   └── actions
└── mask/  # Success indicators
```

## Validation

### Replay Generated Data

Verify data quality:

- Use `scripts/data_tools/visualize_data_zarr.ipynb` to visualize zarr data 

- Use `scripts/mimicgen/replay_no_physics.py` to replay.

```bash
python scripts/mimicgen/replay_demos.py \
    --dataset ./datasets/put_bowl_10k.zarr \
    --num_episodes 10 \
    --render
```

### Data Statistics

Check dataset statistics:

```python
import zarr
data = zarr.open("dataset.zarr", mode="r")
print(f"Total demos: {len(data['data'])}")
print(f"Success rate: {data.attrs['success_rate']}")
```

## Performance Optimization

### GPU Memory

Reduce memory usage:
- Lower `--num_envs` (4-8 recommended)
- Use `--state_only` for prototyping
- Enable `--headless` mode (this must be true for multi-gpu)

### Generation Speed

Improve throughput:
- Increase `--n_procs` with more GPUs
- Use SSD for output directory
- Disable unnecessary observations

### Common Issues

**Out of Memory:**
```bash
# Reduce parallel environments
--num_envs 2
```

**Failed Demos:**
```bash
# Debug with visualization
--num_envs 1 --generation_num_trials 1
# Remove --headless flag
```

