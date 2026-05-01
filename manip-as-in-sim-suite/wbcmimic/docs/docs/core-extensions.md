# Core Extensions

## isaaclab_mimic Extension

The core extension providing robot definitions, task environments, control interfaces, and MimicGen integration.

### Architecture

```
isaaclab_mimic/
├── tasks/              # Task environment definitions
├── mdp/                # MDP components (rewards, terminations)
├── datagen/            # Data generation pipeline
├── wbc/                # Whole-body control
└── utils/              # VR control, IK solvers, robot interfaces
```

### Task Environments

Tasks follow Isaac Lab's manager-based pattern with modular components:

```python
@configclass
class TaskEnvCfg(ManagerBasedEnvCfg):
    # Scene configuration
    scene: SceneCfg
    # Subtask definitions for MimicGen
    subtask_configs: Dict[str, SubTaskConfig]
    # Manager configurations
    observations: ObservationsCfg
    actions: ActionsCfg
    events: EventCfg
```

#### Subtask System

Tasks define subtasks for MimicGen data generation:

```python
subtask_configs = {
    "r": [  # Right arm subtasks (eef names)
        WbcSubTaskConfig(
            name="pick_bowl",
            object_ref="bowl",
            subtask_term_signal="pick_bowl",
            subtask_term_offset_range=(0, 5),
        ),
        WbcSubTaskConfig(
            name="put_bowl_in_microwave",
            object_ref="microwave",
            subtask_term_signal="put_bowl_in_microwave",
            subtask_term_offset_range=(10, 10),
        ),
    ]
}
```

### VR Teleoperation

Collect demonstrations using Meta Quest controllers:

```python
from isaaclab_mimic.utils.vr_policy import DualArmVRPolicy

vr_policy = DualArmVRPolicy()
```

### Control Modes

- **Joint Position**: Direct joint angle control
- **Cartesian IK**: End-effector pose control with IK solver

