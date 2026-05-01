# Installation

- Install IsaacLab 2.1 following the offical [guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)

    - Remove the original `source/isaaclab_mimic` if present

- Install CuRobo following the offical [guide](https://curobo.org/get_started/1_install_instructions.html)

- Download required assets:

```bash
# Download simulation assets from HuggingFace
# These assets should be placed in the source/assets directory
# Available at: https://huggingface.co/datasets/xshenhan/UniMimic

# The assets directory structure should be:   

├── bytemini_assets/        # Simulation environment assets
│   ├── envs/               # Environment models
│   ├── robots/             # Robot models
│   └── objects/            # Interaction objects
└── isaaclab_mimic/         # MimicGen integration
    ├── datagen/            # Data generation pipeline
    ├── tasks/              # Task definitions
    └── wbc/                # Whole-body control algorithms
```

- Install UniMimic extensions:

```bash
cd /path/to/UniMimic
pip install -e source/isaaclab_mimic
```