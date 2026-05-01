# Task Environments API

This page provides API documentation for task environments used in data collection.

## Task Configuration

### Base Configuration

All tasks inherit from `ManagerBasedEnvCfg`:

```python
@configclass
class TaskEnvCfg(ManagerBasedEnvCfg):
    # Scene settings
    scene: SceneCfg
    
    # Subtask definitions for MimicGen
    subtask_configs: Dict[str, List[WbcSubTaskConfig]]
    
    # Manager configurations
    observations: ObservationsCfg
    actions: ActionsCfg
    events: EventCfg
    rewards: RewardsCfg
    terminations: TerminationsCfg
```

### Subtask Configuration

Define subtasks for MimicGen data generation:

```python
@configclass
class WbcSubTaskConfig:
    """Configuration for a subtask."""
    
    # Subtask identification
    name: str
    object_ref: str
    subtask_term_signal: str
    
    # Trajectory selection
    selection_strategy: str = "success"
    num_interpolation_steps: int = 30
    num_fixed_steps: int = 5
    
    # Termination timing
    subtask_term_offset_range: Tuple[int, int] = (10, 15)
    
    # Trajectory clipping
    clip_src_traj_strategy: Optional[str] = None
    slice_to_last_success_step: bool = False
```

## UR5 Tasks

### CloseMicrowave

Simple single-subtask demonstration:

```python
from isaaclab_mimic.tasks.manager_based.ur5_sim import UR5CloseMicrowaveEnvCfg

@configclass
class UR5CloseMicrowaveEnvCfg(UR5BaseConfig):
    """Close microwave door task."""
    
    subtask_configs = {
        "r": [
            WbcSubTaskConfig(
                name="close_door",
                object_ref="microwave",
                subtask_term_signal="close_door",
                subtask_term_offset_range=(20, 25),
            )
        ]
    }
    
    # Scene objects
    microwave: ArticulationCfg = MicrowaveCfg(
        initial_joint_state=0.8  # Door open
    )
```

### PutBowlInMicrowaveAndClose

Complex multi-subtask task:

```python
@configclass
class UR5PutBowlInMicrowaveAndCloseEnvCfg(UR5BaseConfig):
    """Complete microwave task with bowl."""
    
    subtask_configs = {
        "r": [
            WbcSubTaskConfig(
                name="pick_bowl",
                object_ref="bowl",
                subtask_term_signal="pick_bowl",
                subtask_term_offset_range=(20, 25),
            ),
            WbcSubTaskConfig(
                name="put_bowl_in_microwave",
                object_ref="microwave",
                subtask_term_signal="put_bowl_in_microwave",
                subtask_term_offset_range=(10, 15),
                clip_src_traj_strategy="slice_to_last_success_step",
            ),
            WbcSubTaskConfig(
                name="close_door",
                object_ref="microwave",
                subtask_term_signal="close_door",
                subtask_term_offset_range=(10, 15),
            ),
        ]
    }
```

## Task Registration

Tasks must be registered with multiple variants:

```python
# In __init__.py

# Base teleoperation variant
gym.register(
    id="Isaac-UR5-MyTask-Joint-Mimic-v0",
    entry_point="...:UR5MyTaskEnv",
    kwargs={"env_cfg_entry_point": "...:UR5MyTaskMimicEnvCfg"}
)

# Annotation variant
gym.register(
    id="Isaac-UR5-MyTask-Joint-Mimic-Annotate-v0",
    entry_point="...:UR5MyTaskEnv",
    kwargs={"env_cfg_entry_point": "...:UR5MyTaskAnnotateEnvCfg"}
)

# MimicGen generation variant
gym.register(
    id="Isaac-UR5-MyTask-Joint-GoHome-OneCamera-Mimic-MP-v0",
    entry_point="...:UR5MyTaskEnv",
    kwargs={"env_cfg_entry_point": "...:UR5MyTaskMimicGenEnvCfg"}
)
```

## Subtask Signals

Implement subtask completion detection:

```python
class MyTaskEnv(ManagerBasedEnv):
    """Task environment with subtask signals."""
    
    def get_subtask_term_signals(self) -> Dict[str, torch.Tensor]:
        """Compute subtask completion signals."""
        signals = {}
        
        # Check if bowl is picked
        if "pick_bowl" in self.cfg.subtask_signals:
            gripper_closed = self.gripper_state < 0.01
            bowl_in_gripper = self._check_object_in_gripper("bowl")
            signals["pick_bowl"] = gripper_closed & bowl_in_gripper
        
        # Check if bowl is in microwave
        if "put_bowl_in_microwave" in self.cfg.subtask_signals:
            bowl_pos = self.scene["bowl"].data.root_pos_w
            microwave_interior = self._get_microwave_interior_bounds()
            bowl_inside = self._check_position_in_bounds(bowl_pos, microwave_interior)
            signals["put_bowl_in_microwave"] = bowl_inside
        
        # Check if door is closed
        if "close_door" in self.cfg.subtask_signals:
            door_angle = self.scene["microwave"].data.joint_pos[:, self.door_joint_idx]
            signals["close_door"] = door_angle < 0.1  # Nearly closed
        
        return signals
```

## Environment Variants

### Mimic Variant (Teleoperation)

```python
@configclass
class UR5TaskMimicEnvCfg(UR5TaskEnvCfg):
    """Configuration for teleoperation."""
    
    # Enable all subtask signals
    subtask_signals = ["pick_bowl", "put_bowl_in_microwave", "close_door"]
    
    # Dense observations for recording
    observations = ObservationsCfg({
        "policy": {
            "joint_pos": ...,
            "ee_pose": ...,
            "gripper_state": ...,
        }
    })
```

### Annotate Variant

```python
@configclass
class UR5TaskAnnotateEnvCfg(UR5TaskMimicEnvCfg):
    """Configuration for annotation."""
    
    # Disable randomization for consistent replay
    events = EventCfg(
        reset_scene=EventTerm(func=reset_scene_to_default)
    )
    
    # Fixed initial state
    scene.robot.initial_state = FixedInitialState()
```

### MimicGen Variant

```python
@configclass
class UR5TaskMimicGenEnvCfg(UR5TaskEnvCfg):
    """Configuration for data generation."""
    
    # Add cameras for visual observations
    scene.camera_wrist = CameraCfg(
        prim_path="/World/envs/env_.*/robot/wrist_camera",
        resolution=(640, 480),
        focal_length=24.0,
    )
    
    # Optimized for parallel generation
    sim.dt = 0.01  # Faster simulation
    decimation = 2  # Less frequent policy updates
```

## Common Patterns

### Object Spawning

```python
@configclass
class ObjectCfg(RigidObjectCfg):
    """Spawnable object configuration."""
    
    # Spawn parameters
    spawn_height_range: Tuple[float, float] = (0.0, 0.02)
    spawn_xy_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.1, -0.1), (0.1, 0.1))
    
    # Initial state
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.1),
        rot=(1.0, 0.0, 0.0, 0.0),
    )
```

### Observation Terms

```python
def joint_pos_rel(env: ManagerBasedEnv) -> torch.Tensor:
    """Joint positions relative to default."""
    return env.scene.robot.data.joint_pos - env.scene.robot.data.default_joint_pos

def ee_pose(env: ManagerBasedEnv) -> torch.Tensor:
    """End-effector pose as position + quaternion."""
    ee_state = env.scene.robot.data.ee_state
    return torch.cat([ee_state[:, :3], ee_state[:, 3:7]], dim=-1)

def gripper_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Gripper opening (0=closed, 1=open)."""
    gripper_dof_idx = env.scene.robot.gripper_dof_indices
    return env.scene.robot.data.joint_pos[:, gripper_dof_idx]
```

### Reward Terms

```python
def reaching_reward(env: ManagerBasedEnv, target_name: str, sigma: float = 0.1) -> torch.Tensor:
    """Exponential reward for reaching target."""
    ee_pos = env.scene.robot.data.ee_state[:, :3]
    target_pos = env.scene[target_name].data.root_pos_w
    distance = torch.norm(ee_pos - target_pos, dim=-1)
    return torch.exp(-distance / sigma)

def grasping_reward(env: ManagerBasedEnv, threshold: float = 5.0) -> torch.Tensor:
    """Reward for successful grasping."""
    contact_forces = env.scene.robot.data.net_contact_forces
    gripper_forces = contact_forces[:, env.scene.robot.gripper_link_indices].sum(dim=1)
    return (gripper_forces > threshold).float()
```

## Usage Examples

### Creating a Task

```python
# Load configuration
from isaaclab_mimic.tasks.manager_based.ur5_sim import UR5CloseMicrowaveEnvCfg

# Create environment
env_cfg = UR5CloseMicrowaveEnvCfg()
env_cfg.scene.num_envs = 16
env = ManagerBasedEnv(cfg=env_cfg)

# Run simulation
obs, _ = env.reset()
for _ in range(100):
    actions = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(actions)
```

### Recording Demonstrations

```python
from isaaclab_mimic.utils.vr_policy import DualArmVRPolicy

# Setup VR policy
vr_policy = DualArmVRPolicy(robot_name="ur5")

# Record demonstrations
demos = []
for episode in range(10):
    obs, _ = env.reset()
    trajectory = []
    
    while not done:
        actions = vr_policy.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(actions)
        trajectory.append({"obs": obs, "action": actions})
        done = terminated or truncated
    
    demos.append(trajectory)
```