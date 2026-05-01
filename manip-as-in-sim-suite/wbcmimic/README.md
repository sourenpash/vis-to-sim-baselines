# WBCMimic

WBCMimic is an enhanced version of [MimicGen](https://github.com/NVlabs/mimicgen) that extends autonomous data generation to mobile manipulators with whole-body control (WBC). It enables efficient generation of high-quality manipulation demonstrations through automated data generation pipelines with multi-GPU parallel simulation support.

## ✨ Key Features

- **Whole-Body Control**: Unified control for mobile manipulators including base motion
- **Automated Data Generation**: Scalable demonstration generation using MimicGen principles  
- **Multi-GPU Parallelization**: Distributed simulation across multiple GPUs for faster data collection
- **VR Teleoperation**: Intuitive demonstration recording using Meta Quest controllers
- **Smooth Motion Generation**: Improved trajectory smoothness compared to original MimicGen
- **Flexible Task Definition**: Support for complex multi-stage manipulation tasks

## 📋 Prerequisites

- [Isaac Lab 2.1](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)
- [CuRobo](https://curobo.org/get_started/1_install_instructions.html) for motion planning
- [Oculus Reader](https://github.com/rail-berkeley/oculus_reader) (for VR teleoperation)

## 🛠️ Installation

1. **Install Isaac Lab 2.1** following the [official guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)
   - Note: Remove the original `source/isaaclab_mimic` if present

2. **Install CuRobo** following the [official guide](https://curobo.org/get_started/1_install_instructions.html)

3. **Download required assets**:
   ```bash
   # Download simulation assets from HuggingFace
   # Assets available at: https://huggingface.co/datasets/xshenhan/UniMimic
   # Place assets in the appropriate source/wbcmimic_assets directory
   
   # Expected directory structure:
   source/
   ├── wbcmimic_assets/                 # Simulation environment assets
   │   ├── UR5/                      # Robot models (UR5, ARX-X7)
   │   ├── objects/                     # Manipulation objects
   │   └── environments/                # Environment models
   │   └── ...                          
   └── isaaclab_mimic/isaaclab_mimic/   # WBCMimic core package
       ├── datagen/                     # Data generation pipeline
       ├── tasks/                       # Task definitions
       ├── envs/                        # Environment configurations
       └── ...                          # Other files and directories
   ```

4. **Install WBCMimic package**:
   ```bash
   cd wbcmimic
   pip install -e source/isaaclab_mimic
   ```

## 🚦 Quick Start

WBCMimic uses a three-step workflow for generating large-scale demonstration data:

### Step 1: Record Demonstrations

Record initial human demonstrations using VR teleoperation:

> **Requirements**: Meta Quest 3 headset and [Oculus reader](https://github.com/rail-berkeley/oculus_reader) for data recording.

```bash
python scripts/basic/record_demos_ur5_quest.py \
    --task Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-v0 \
    --dataset_file ./demos/clean_plate.hdf5 \
    --num_demos 5
```

**VR Controls:**
- **Quest Controllers**: Control robot end-effector position and orientation
- **R key**: Reset environment and discard current recording
- **C key**: Continue/resume recording current episode

### Step 2: Annotate Subtasks

Add subtask completion annotations to enable MimicGen data generation:

```bash
python scripts/mimicgen/annotate_demos.py \
    --task Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-Annotate-v0 \
    --input_file ./demos/clean_plate.hdf5 \
    --output_file ./demos/clean_plate_annotated.hdf5 \
    --auto
```

### Step 3: Generate Large-Scale Data

Use WBCMimic's parallelized MimicGen to generate thousands of demonstrations automatically:

```bash
python scripts/mimicgen/generate_dataset_parallel_all.py \
    --task Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-MP-v0 \
    --input_file ./demos/clean_plate_annotated.hdf5 \
    --output_file ./datasets/clean_plate_10k.zarr \
    --generation_num_trials 125 \
    --num_envs 4 \
    --n_procs 8 \
    --distributed \
    --enable_cameras \
    --mimic_mode uni \
    --headless
```

This generates 1,000 demonstrations (125 trials × 8 processes) using distributed parallel simulation.

**Note**: Use `--mimic_mode wbc` for whole-body control or `--mimic_mode uni` for unified control mode.

## 📚 Examples

### Example: UR-Put-Bowl-In-MicroWave-And-Close-Door

Record and generate data for a complex manipulation task:

```bash
# Record demonstrations for putting bowl in microwave and closing it
python scripts/basic/record_demos_ur5_quest.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-Mimic-v0 \
    --dataset_file ./demos/bowl_microwave.hdf5 \
    --num_demos 10

# Annotate the subtasks
python scripts/mimicgen/annotate_demos.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-Mimic-Annotate-v0 \
    --input_file ./demos/bowl_microwave.hdf5 \
    --output_file ./demos/bowl_microwave_annotated.hdf5 \
    --auto

# Generate large-scale dataset
python scripts/mimicgen/generate_dataset_parallel_all.py \
    --task Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-OneCamera-Mimic-MP-v0 \
    --input_file ./demos/bowl_microwave_annotated.hdf5 \
    --output_file ./datasets/bowl_microwave_50k.zarr \
    --generation_num_trials 125 \
    --num_envs 4 \
    --n_procs 8 \
    --distributed \
    --enable_cameras \
    --mimic_mode uni \
    --headless

# With hydra configuration

python scripts/mimicgen/run_hydra.py --config-name config/user/generation/generate_ur5_clean_plate_715_high_arm.yaml
```

## 🗂️ Project Structure

```
wbcmimic/
├── source/
│   └── isaaclab_mimic/        # Core WBCMimic package
│       ├── datagen/           # Data generation pipeline with parallelization
│       ├── tasks/             # Task definitions (UR5, ARX-X7)
│       ├── envs/              # Environment configurations
│       ├── utils/             # Utilities including WBC controllers
│       └── managers/          # Event and RL managers
├── scripts/
│   ├── basic/                 # VR demonstration recording
│   ├── mimicgen/              # Data generation and replay
│   ├── data_tools/            # Data processing utilities
│   └── tools/                 # Additional helper scripts
├── config/                    # Hydra configuration files
│   ├── mimicgen/              # Core MimicGen configs
│   ├── user/                  # User-specific task configs
│   └── logging/               # Logging configurations
├── docs/                      # Documentation and guides
└── install.sh                 # Installation script
```

## 📖 Documentation

Build and serve the complete documentation locally:

```bash
cd docs
pip install mkdocs mkdocs-material "mkdocstrings[python]" pymdown-extensions
mkdocs serve 
# Or use the build script:
./build_docs.sh
```

Visit [http://localhost:8000](http://localhost:8000) to view the documentation.

## 🎯 Supported Tasks

| Task Name | Task ID | Robot | Subtasks | Description |
|-----------|---------|-----------|----------|-------------|
| Put Bowl in Microwave & Close | `Isaac-UR5-PutBowlInMicroWaveAndClose-*` | UR5-Robotiq | 3 | Pick bowl, place in microwave, close door |
| Clean Plate | `Isaac-UR5-CleanPlate-*` | UR5-Robotiq | 6 | Fork pickup, cleaning, plate manipulation with retry logic |
| Pick Toothpaste Into Cup & Push | `Isaac-X7-PickToothpasteIntoCupAndPush-*` | ARX-X7 | 3 | Dual-arm mobile manipulation with base motion |

**Task Variants**: Each task supports multiple control modes:
- `*Joint*`: Joint space control with WBC
- `*EE*`: End-effector control  
- `*MP*`: Multi-process for parallel data generation
- `*OneCamera*`: Single camera observation
- `*GoHome*`: Return to home position
- `*Retry*`: Include retry logic for failed placements
- `*Annotate*`: For replay and trajectory annotation
- `*Wbc*`: Explicit whole-body control mode
- `*NoCrop*`: Uncropped camera observations

## 🔧 Configuration

WBCMimic uses [Hydra](https://hydra.cc/) for flexible configuration management. All configurations are located in the `config/` directory:

### Configuration Structure

```
config/
├── logging/                    # Logging configurations
│   └── disabled.yaml          # Disable all logging
├── mimicgen/                  # Core MimicGen configurations
│   ├── generate_data.yaml    # Base data generation template
│   └── replay_no_physics.yaml # Replay demonstrations without physics
└── user/                     # Task-specific configurations
    ├── generation/           # Data generation configs
    └── replay/              # Replay configs
```

## 📝 License

This project is licensed under the Apache 2.0 License. See [LICENSE](../LICENSE) for details.

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Test data generation pipeline
python source/isaaclab_mimic/test/test_generate_dataset.py

# Test selection strategies
python source/isaaclab_mimic/test/test_selection_strategy.py

# Test robot IK (Jupyter notebook)
jupyter notebook source/isaaclab_mimic/test/test_robot_ik.ipynb
```

## 🔗 Links

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [CuRobo Motion Planning](https://curobo.org/) 
- [Oculus Reader for VR](https://github.com/rail-berkeley/oculus_reader)
- [MimicGen Original](https://github.com/NVlabs/mimicgen)
- [WBCMimic Assets](https://huggingface.co/datasets/xshenhan/UniMimic)

## 📄 Citation

If you use WBCMimic in your research, please cite:

```bibtex
@article{liu2025manipulation,
  title={Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots},
  author={Liu, Minghuan and Zhu, Zhengbang and Han, Xiaoshen and Hu, Peng and Lin, Haotong and 
          Li, Xinyao and Chen, Jingxiao and Xu, Jiafeng and Yang, Yichu and Lin, Yunfeng and 
          Li, Xinghang and Yu, Yong and Zhang, Weinan and Kong, Tao and Kang, Bingyi},
  journal={arXiv preprint},
  year={2025}
}
```
