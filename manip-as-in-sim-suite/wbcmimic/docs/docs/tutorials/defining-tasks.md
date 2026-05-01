# Defining Custom Tasks

This tutorial shows how to create custom tasks for UniMimic's MimicGen data generation. A task requires three components: subtask termination logic, observation configuration, and environment implementation.

## 1. Defining Subtask Termination Logic

Subtask success detection is critical for data collection quality. Poor conditions lead to low success rates.

### Key Principles

- **Use compound conditions**: Combine multiple checks to avoid false positives
- **Handle edge cases**: Account for physics instabilities and temporary states  
- **Tune thresholds carefully**: Balance between robustness and precision

### Example: Pick and Place Task

For picking objects, combine lift detection with grasp verification:

```python
pick_bowl = ObsTerm(
    func=mdp.and_func,
    params={
        "funcs": {"lift": mdp.lift_success_simple, "grasp": mdp.is_object_grasped},
        "kwargs": {
            "lift": {
                "item_cfg": SceneEntityCfg("bowl"),
                "height_threshold": 0.01,  # Small threshold for robustness
            },
            "grasp": {
                "robot_grasp_cfg": SceneEntityCfg("robot", 
                    body_names=["ur_robotiq_85_left_finger_tip_link", 
                               "ur_robotiq_85_right_finger_tip_link"]),
            }
        }
    },
)
```

For placement, ensure robot moved away and object is positioned correctly:

```python
put_bowl_in_microwave = ObsTerm(
    func=mdp.and_func,
    params={
        "funcs": {"robot_away": mdp.stay_away, "object_placed": mdp.a_in_b()},
        "kwargs": {
            "robot_away": {   
                "obj_1_cfg": "tcp_transform",
                "obj_2_cfg": SceneEntityCfg("bowl"),
                "threshold": 0.12,
            },
            "object_placed": {
                "a": SceneEntityCfg("bowl"),
                "b": SceneEntityCfg("microwave", joint_names=["Joints"]),
            }
        }
    },
)
```

For articulated objects like doors:

```python
close_door = ObsTerm(
    func=mdp.close_articulation,
    params={
        "item_entity": SceneEntityCfg("microwave", joint_names=["Joints"]),
        "threshold": (-0.05, 0.1),  # Joint position range for "closed"
    },
)
```

## 2. Observation Configuration

Subtask signals must be exposed through observations for MimicGen to access them.

Create a `SubtaskCfg` observation group:

```python
@configclass
class ObservationsCfg(UR5ObservationsMimicCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        pick_bowl = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"lift": mdp.lift_success_simple, "grasp": mdp.is_object_grasped},
                "kwargs": {
                    "lift": {"item_cfg": SceneEntityCfg("bowl"), "height_threshold": 0.01},
                    "grasp": {"robot_grasp_cfg": SceneEntityCfg("robot", 
                        body_names=["ur_robotiq_85_left_finger_tip_link", 
                                   "ur_robotiq_85_right_finger_tip_link"])}
                }
            },
        )
        
        put_bowl_in_microwave = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"robot_away": mdp.stay_away, "object_placed": mdp.a_in_b()},
                "kwargs": {
                    "robot_away": {   
                        "obj_1_cfg": "tcp_transform",
                        "obj_2_cfg": SceneEntityCfg("bowl"),
                        "threshold": 0.12,
                    },
                    "object_placed": {
                        "a": SceneEntityCfg("bowl"),
                        "b": SceneEntityCfg("microwave", joint_names=["Joints"]),
                    }
                }
            },
        )

        close_door = ObsTerm(
            func=mdp.close_articulation,
            params={
                "item_entity": SceneEntityCfg("microwave", joint_names=["Joints"]),
                "threshold": (-0.05, 0.1),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()
```

**Important**: 
- Observation term names must exactly match `subtask_term_signal` in subtask configs
- Always set `enable_corruption = False` and `concatenate_terms = False`

## 3. Environment Implementation

Inherit from the appropriate base class and implement `get_subtask_term_signals`:

```python
from isaaclab_mimic.tasks.manager_based.ur5_sim.envs.ur5_joint_mimic_env import UR5BaseMimicJointControlEnv

class MyTaskMimicJointControlEnv(UR5BaseMimicJointControlEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Returns subtask termination signals for MimicGen."""
        if env_ids is None:
            env_ids = slice(None)
            
        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        
        # Map each subtask signal (must match observation term names)
        signals["pick_bowl"] = subtask_terms["pick_bowl"][env_ids]
        signals["put_bowl_in_microwave"] = subtask_terms["put_bowl_in_microwave"][env_ids]
        signals["close_door"] = subtask_terms["close_door"][env_ids]
        
        return signals
```

### Base Class Options

- `UR5BaseMimicJointControlEnv`: For WBC with joint control
- `UR5BaseMimicEEControlEnv`: For direct end-effector control

## 4. Complete Task Configuration

Connect everything together:

```python
@configclass
class MyTaskMimicEnvCfg(MimicEnvCfg, ManagerBasedRLEnvCfg):
    def __post_init__(self) -> None:
        # Scene and observations
        self.scene = MyTaskSceneCfg(num_envs=2, env_spacing=40.0)
        self.observations = ObservationsCfg()
        
        # Define subtask sequence
        subtask_configs = []
        
        subtask_configs.append(
            WbcSubTaskConfig(
                object_ref="bowl",
                subtask_term_signal="pick_bowl",  # Must match observation term
                subtask_term_offset_range=(0, 10),
                wbc_max_step=5,
                wbc_max_step_start=100,
            )
        )
        
        subtask_configs.append(
            WbcSubTaskConfig(
                object_ref="microwave",
                subtask_term_signal="put_bowl_in_microwave",
                subtask_term_offset_range=(0, 10),
                wbc_max_step=5,
                wbc_max_step_start=100,
            )
        )
        
        subtask_configs.append(
            WbcSubTaskConfig(
                object_ref="microwave",
                subtask_term_signal="close_door",
                subtask_term_offset_range=(0, 0),
                slice_to_last_success_step=True,
                subtask_continous_step=1,  # Require stable closure
            )
        )
        
        self.subtask_configs["r"] = subtask_configs
```

By following these steps, you can create robust tasks that collect high-quality demonstration data.