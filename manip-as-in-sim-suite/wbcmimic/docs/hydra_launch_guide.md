# Hydra Launch Configuration Guide

This guide explains how to use Hydra for launching data generation scripts in the UniMimic project.

## Overview

The project uses [Hydra](https://hydra.cc/) for configuration management, allowing flexible and modular configuration of data generation tasks. The main entry point is `scripts/mimicgen/run_hydra.py`, which wraps the `generate_dataset_parallel_all.py` script with Hydra configuration support.

## Configuration Structure

### Directory Layout

```
UniMimic/config/
├── logging/
│   └── disabled.yaml      # Disable logging
├── mimicgen/
│   ├── generate_data.yaml # Main data generation config
│   └── replay_no_physics.yaml
└── user/
    ├── generation/        # User-specific generation configs
    └── replay/           # User-specific replay configs
```

### Main Configuration Files

#### 1. Base Configuration (`config/mimicgen/generate_data.yaml`)

This is the main configuration file that defines all available parameters:

```yaml
# @package _global_
defaults:
  - ../logging@logging: swanlab  # Default logging configuration
  - _self_

# Task configuration
task: null                      # Task name (e.g., "Isaac-Lift-Cube-Franka-v0")
generation_num_trials: 100      # Number of demos to generate
num_envs: 1                    # Number of parallel environments

# File paths
input_file: null               # Source dataset file (required)
output_file: "./datasets/output_dataset.hdf5"  # Output file path

# Generation options
pause_subtask: false           # Debug: pause after each subtask
distributed: false             # Use multiple GPUs/nodes
n_procs: 8                    # Number of processes
state_only: false             # Only save state information
record_all: false             # Record all data including failures
seed: 42                      # Random seed

# Control mode
mimic_mode: "wbc"             # Options: "origin", "wbc", "uni"

# Episode saving
failed_save_prob: 0.1         # Probability to save failed episodes

# Batch splitting
batch_splits: null            # Split into batches, e.g., [100, 200, 300]

# File transfer configuration
file_transfer:
  local_base_dir: "/path/to/local/datasets/"
  remote_base_dir: "/path/to/remote/datasets/"
  transfers: 48               # Number of parallel transfers
  delete_after_transfer: false # Delete local files after transfer

# Monitoring
monitoring:
  timeout: 300               # Warning timeout (seconds)
  check_interval: 20         # Check interval (seconds)

# Extra arguments passed to the script
extra_args:
  enable_cameras: true
  distributed: true
  headless: true
  n_procs: 8

# WBC solver configuration (optional)
wbc_solver_cfg: null
```

#### 2. Logging Configuration (`config/logging/disabled.yaml`)

```yaml
enabled: false
    secret: "your_secret_key"
```

#### 3. User Configuration Example

Create custom configurations by extending the base configuration:

```yaml
# @package _global_
defaults:
  - ../../mimicgen@: generate_data    # Inherit from base config
  - ../../logging@logging: swanlab    # Use SwanLab logging

# Override specific parameters
task: Isaac-UR5-CleanPlate-Mimic-v0
input_file: /path/to/source_dataset.hdf5
output_file: /path/to/output_dataset.zarr
generation_num_trials: 125
num_envs: 8
seed: 2025

# Custom batch splitting
batch_splits: [100, 100, 100, 100, 100]

# Override monitoring timeout
monitoring:
  timeout: 600

# Enable file deletion after transfer
file_transfer:
  delete_after_transfer: true
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python scripts/mimicgen/run_hydra.py

# Specify task and input file
python scripts/mimicgen/run_hydra.py \
    task=Isaac-Lift-Cube-Franka-v0 \
    input_file=/path/to/source.hdf5 \
    generation_num_trials=200
```

### Using Custom Configuration

```bash
# Use a predefined user configuration
python scripts/mimicgen/run_hydra.py \
    --config-name config/user/generation/generate_ur5_clean_plate_75_high_arm

# Override specific parameters
python scripts/mimicgen/run_hydra.py \
    --config-name config/user/generation/generate_ur5_clean_plate_75_high_arm \
    generation_num_trials=500 \
    seed=123
```

### Disable Logging

```bash
python scripts/mimicgen/run_hydra.py logging=disabled
```

## Key Features

### 1. Batch Processing

The system supports splitting data generation into batches for better management:

```yaml
batch_splits: [100, 200, 300, 400]  # Generate 1000 episodes in 4 batches
```

Each batch is saved as a separate file and can be transferred independently.

### 2. Automatic File Transfer

Files can be automatically transferred to remote storage using rclone:

```yaml
file_transfer:
  local_base_dir: "/local/path/"
  remote_base_dir: "/remote/path/"
  transfers: 48
  delete_after_transfer: true  # Clean up local files
```

### 3. Real-time Monitoring

- SwanLab integration for experiment tracking
- Lark/Feishu notifications for important events
- Timeout warnings if episodes take too long

### 4. Control Modes

Three control modes are supported:
- `origin`: Original control mode
- `wbc`: Whole-body control
- `uni`: Unified control

### 5. WBC Solver Configuration

For advanced users, custom WBC solver parameters can be specified:

```yaml
wbc_solver_cfg:
  parameter1: value1
  parameter2: value2
```

## Command Line Override Syntax

Hydra supports various ways to override configuration:

```bash
# Override single value
python run_hydra.py seed=123

# Override nested value
python run_hydra.py file_transfer.transfers=64

# Override list
python run_hydra.py 'batch_splits=[100,200,300]'

# Add to extra_args
python run_hydra.py +extra_args.new_param=value
```

## Creating New Configurations

1. Create a new YAML file in `config/user/generation/`
2. Use the `defaults` list to inherit from base configurations
3. Override only the parameters you need to change
4. Reference it using `--config-name` when running

Example structure:
```yaml
# @package _global_
defaults:
  - ../../mimicgen@: generate_data
  - ../../logging@logging: disabled  # Disable logging for this config

# Your custom overrides here
task: Your-Custom-Task
generation_num_trials: 1000
```

## Troubleshooting

1. **Configuration not found**: Ensure the config path is relative to the `config/` directory
2. **Parameter override not working**: Check that you're using the correct nested syntax
3. **Batch files not transferring**: Verify rclone is configured and the paths are correct
4. **Timeout warnings**: Increase `monitoring.timeout` for slow tasks

## Best Practices

1. Create user-specific configurations instead of modifying base configs
2. Use batch splitting for large datasets to enable incremental transfers
3. Enable `delete_after_transfer` to save disk space
4. Set appropriate timeouts based on your task complexity
5. Use meaningful configuration names that describe the task