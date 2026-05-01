# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_mimic.utils.path import PROJECT_ROOT
from isaaclab_mimic.utils.rigid_body import RigidObjectInitDataAuto

from . import mdp


@configclass
class CleanPlateSceneCfg(InteractiveSceneCfg):
    """Configuration for a clean plate environment."""

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.25),
            rot=(1.0, 0, 0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/simple_objects/greentable2.usdc",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            scale=(0.799, 1.198, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.6)),
        spawn=GroundPlaneCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/default_environment.usd"
        ),
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )

    robot: ArticulationCfg = MISSING
    # arm_camera: CameraCfg|TiledCameraCfg = MISSING
    # third_camera: CameraCfg|TiledCameraCfg = MISSING
    # third_camera2: CameraCfg|TiledCameraCfg = MISSING

    left_blanket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/left_blanket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.20, 0.22, 0.05),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/objects/plastic_box/plastic_box.usdc",
            mass_props=sim_utils.MassPropertiesCfg(
                mass=100,
            ),
            # scale=(1.343, 1, 1)
        ),
    )

    right_blanket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/right_blanket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.12, 0.0353, 0.05),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/objects/plastic_box/plastic_box.usdc",
            mass_props=sim_utils.MassPropertiesCfg(
                mass=100,
            ),
            # scale=(1.343, 1, 1)
        ),
    )

    plate = RigidObjectCfg(
        class_type=RigidObjectInitDataAuto,
        prim_path="{ENV_REGEX_NS}/plate",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.26, -0.04, 0.04), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/simple_objects/plate/plate.usdc",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                disable_gravity=False,
                max_depenetration_velocity=100,  # avoid penetration??
            ),
            scale=(1.0, 1.0, 0.9),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.04,
            ),
        ),
    )

    fork = RigidObjectCfg(
        class_type=RigidObjectInitDataAuto,
        prim_path="{ENV_REGEX_NS}/fork",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.1, -0.04, 0.05), rot=(0, 1, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/objects/fork/fork.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                disable_gravity=False,
                max_depenetration_velocity=100,  # avoid penetration??
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.001,
            ),
        ),
    )

    # chicken = RigidObjectCfg(
    #     class_type=RigidObjectInitDataAuto,
    #     prim_path="{ENV_REGEX_NS}/chicken",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(-0.1, -0.07, 0.08), rot=(1, 0, 0, 0)
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/objects/fried_chicken/scene.usdc",
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             disable_gravity=False,
    #             max_depenetration_velocity=100,  # avoid penetration??
    #         ),
    #         scale=(0.01, 0.01, 0.01),
    #         mass_props=sim_utils.MassPropertiesCfg(
    #             mass=0.001,
    #         )
    #     ),
    # )


@configclass
class EventCfg:
    reset_plate = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("plate"),
            "pose_range": {
                "x": (-0.1, 0.13),
                "y": (-0.2, 0.14),
                # "x": (-0.1, 0.05),
                # "y": (-0.2, 0.1),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {},
        },
        min_step_count_between_reset=10,
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {},
            "velocity_range": {},
        },
    )


@configclass
class _TimeoutTerminationsCfg:
    timeout = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # success = DoneTerm(
    #     func=mdp.close_articulation,
    #     time_out=False,
    #     params={
    #         "item_entity": SceneEntityCfg("microwave", joint_names=["Joints"]),
    #         "threshold": (-0.1, 0.1),
    #     },
    # )

    early_stop = DoneTerm(
        func=mdp.item_out_of_bounds,
        time_out=False,
        params={
            "item_cfg": SceneEntityCfg("plate"),
            "bounds": {
                "x": (-0.4, 0.4),
                "y": (-0.6, 0.6),
                "z": (-0.1, 10.0),
            },
        },
    )

    # success = DoneTerm(
    #     func=mdp.and_func,
    #     time_out=False,
    #     params={
    #         "funcs": {"a": mdp.close_articulation, "b": mdp.a_in_b()},
    #         "kwargs": {
    #             "a": {
    #                 "item_entity": SceneEntityCfg("microwave", joint_names=["Joints"]),
    #                 "threshold": (-0.1, 0.1),
    #             },
    #             "b": {
    #                 "a":  SceneEntityCfg("bowl"),
    #                 "b":  SceneEntityCfg("microwave", joint_names=["Joints"]),
    #             }
    #         }
    #     },
    # )

    success = DoneTerm(
        func=mdp.success_all_task_sequentially,
        time_out=False,
        params={},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=0.01)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    pass


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pass


if __name__ == "__main__":
    scene_cfg = CleanPlateSceneCfg()
    print(scene_cfg)
