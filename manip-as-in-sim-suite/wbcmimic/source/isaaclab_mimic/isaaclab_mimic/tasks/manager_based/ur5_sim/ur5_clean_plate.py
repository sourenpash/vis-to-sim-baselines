# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    CameraCfg,
    ContactSensorCfg,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCameraCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_mimic.envs import (
    KeyPointSubTaskConfig,
    ManagerBasedRLMimicEnv,
    WbcSubTaskConfig,
)
from isaaclab_mimic.envs.clip_src_traj_strategy import LastStepClipCfg
from isaaclab_mimic.managers.mimic_event_manager import (
    MimicEventTermCfg as MimicEventTerm,
)
from isaaclab_mimic.managers.mimic_event_manager import TriggerBase

# ours
from isaaclab_mimic.utils.path import PROJECT_ROOT
from isaaclab_mimic.utils.robots.wbc_controller import WbcControllerCfg

from . import mdp
from .clean_plate import (
    CleanPlateSceneCfg,
    CommandsCfg,
    EventCfg,
    RewardsCfg,
    _TimeoutTerminationsCfg,
)
from .robots import (
    UR5_CFG,
    UR5ActionsCartesianCfg,
    UR5JointsActionsCfg,
    UR5MimicCfg,
    UR5ObservationsMimicCfg,
)


@configclass
class UR5CleanPlateSceneRobotTableCfg(CleanPlateSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot_table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/robot_table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-0.65, 0.277, -0.42175),
                rot=(1.0, 0, 0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/simple_objects/greentable.usdc",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                scale=(0.49, 0.65, 0.74372),
            ),
        )
        self.bg_table_front = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/bg_table_front",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0, 0.62552, -5),
                rot=(1.0, 0, 0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/simple_objects/greentable.usdc",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                scale=(4, 0.01, 12.5),
            ),
        )
        self.bg_table_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/bg_table_right",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.49611, 0, -3),
                rot=(1.0, 0, 0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/simple_objects/greentable.usdc",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                scale=(0.01, 2.4, 8),
            ),
        )


@configclass
class UR5CleanPlateSceneCfg(UR5CleanPlateSceneRobotTableCfg):
    """Configuration for the UR5 clean plate environment."""

    third_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/third_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        # spawn=sim_utils.PinholeCameraCfg( # 326.28186035 240.93412781
        #     clipping_range=(0.001, 3),
        #     focal_length=34.048,  # mm, approx fx-mapped
        #     focus_distance=400.0,  # arbitrary
        #     horizontal_aperture=36.0,  # mm
        #     vertical_aperture=24.0,    # mm
        # ),
        # Try to Match Realsense D435i (25.07.04)
        spawn=sim_utils.PinholeCameraCfg(  # 326.28186035 240.93412781
            clipping_range=(0.001, 3.0),
            focal_length=3.769,  # mm, approx fx-mapped
            focus_distance=400.0,  # arbitrary
            horizontal_aperture=3.984,  # mm
            vertical_aperture=2.952,  # mm
            # horizontal_aperture_offset=-0.0391,
            # vertical_aperture_offset=-0.0057,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # pos=(-0.1556869,  -0.42937925,  1.04282238), # new diff calibrate
            # rot=(0.21699706, -0.97591795,  0.01074349, -0.01951953),
            # rot=(0.21767628, -0.97498268,  0.03120586, -0.0324345), # old
            # pos=(-0.160805,   -0.43487581,  1.0508559),
            pos=(-0.15281393, -0.42782652, 1.04246988),  # 20250715
            rot=(0.21607311, -0.976270928, 0.000276836672, -0.0144017304),
            convention="ros",
        ),
    )

    l515_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/l515_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(  # 326.28186035
            clipping_range=(0.001, 3.0),
            focal_length=19.86,  # mm, approx fx-mapped
            focus_distance=400.0,  # arbitrary
            horizontal_aperture=20.955,  # mm
            vertical_aperture=15.716,  # mm
            # horizontal_aperture_offset=-0.3113,
            # vertical_aperture_offset=-0.1201,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0233195, -0.4159497, 0.94999548),
            rot=(0.24775141, -0.96474501, -0.08827128, -0.00972034),
            convention="ros",
        ),
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.robot = UR5_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-6.25179097e-01, 4.98731775e-02, -4.44625377e-02),
                rot=(1, 0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan_joint": -3.5580e00,
                    "shoulder_lift_joint": -1.3595e00,
                    "elbow_joint": -2.1516e00,
                    "wrist_1_joint": -1.0714e00,
                    "wrist_2_joint": 1.7125e00,
                    "wrist_3_joint": -4.1658e-01,
                    "gripper_joint": 0.0,
                    "ur_robotiq_85_.*": 0.0,
                },
            ),
        )

        self.tcp_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                    name="tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.14), rot=(1.0, 0.0, 0.0, 0.0)),
                )
            ],
            # debug_vis=True,
        )

        self.ur_robotiq_85_left_finger_tip_link_cf = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur_robotiq_85_left_finger_tip_link",
            update_period=0.0,
            history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/plate/*", "{ENV_REGEX_NS}/fork/*"],
        )
        self.ur_robotiq_85_right_finger_tip_link_cf = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur_robotiq_85_right_finger_tip_link",
            update_period=0.0,
            history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/plate/*", "{ENV_REGEX_NS}/fork/*"],
        )


@configclass
class OneCameraObs(UR5ObservationsMimicCfg):
    """Observations for the UR5 clean plate environment with one camera."""

    @configclass
    class ImageCfg(ObsGroup):
        """Observations for policy group."""

        camera_0 = ObsTerm(
            func=mdp.get_camera_image,
            params={
                "camera_entity": SceneEntityCfg("third_camera"),
            },
        )
        camera_1 = ObsTerm(
            func=mdp.get_camera_image,
            params={
                "camera_entity": SceneEntityCfg("l515_camera"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraInfoCfg(ObsGroup):
        camera_0_intr = ObsTerm(
            func=mdp.get_camera_intr,
            params={
                "camera_entity": SceneEntityCfg("third_camera"),
            },
        )

        camera_0_extr = ObsTerm(
            func=mdp.get_camera_extr,
            params={
                "camera_entity": SceneEntityCfg("third_camera"),
            },
        )

        camera_1_intr = ObsTerm(
            func=mdp.get_camera_intr,
            params={
                "camera_entity": SceneEntityCfg("l515_camera"),
            },
        )

        camera_1_extr = ObsTerm(
            func=mdp.get_camera_extr,
            params={
                "camera_entity": SceneEntityCfg("l515_camera"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DepthCfg(ObsGroup):
        camera_0 = ObsTerm(
            func=mdp.get_camera_depth,
            params={
                "camera_entity": SceneEntityCfg("third_camera"),
                "resize": True,
                "resize_shape": (240, 320),  # Resize to 240x320
            },
        )

        camera_1 = ObsTerm(
            func=mdp.get_camera_depth,
            params={
                "camera_entity": SceneEntityCfg("l515_camera"),
                "resize": True,
                "resize_shape": (240, 320),  # Resize to 240x320
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PolicyInferenceObsCfg(ObsGroup):
        pointcloud = ObsTerm(
            func=mdp.get_pointcloud_obs,
            params={
                "cameras": [SceneEntityCfg("third_camera")],
                "crop_bound": [0.2, 1.21, -1.2, 1.2, -0.3, 0.7],
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        pick_fork = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"a": mdp.lift_success_simple, "b": mdp.approach_success},
                "kwargs": {
                    "a": {
                        "item_cfg": SceneEntityCfg("fork"),
                        "height_threshold": 0.06,
                    },
                    "b": {
                        "obj_1_cfg": "tcp_transform",
                        "obj_2_cfg": SceneEntityCfg("fork"),
                        "threshold": 0.1,
                    },
                },
            },
        )

        place_fork = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"a": mdp.stay_away, "b": mdp.a_in_b()},
                "kwargs": {
                    "a": {
                        "obj_1_cfg": "tcp_transform",
                        "obj_2_cfg": SceneEntityCfg("fork"),
                        "threshold": 0.08,
                    },
                    "b": {
                        "a": SceneEntityCfg("fork"),
                        "b": SceneEntityCfg(
                            "right_blanket"
                        ),  # Using fork as target for cleaning task
                        "ratio": 0.2,
                    },
                },
            },
        )

        pick_plate = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"a": mdp.lift_success_simple, "b": mdp.is_object_grasped},
                "kwargs": {
                    "a": {
                        "item_cfg": SceneEntityCfg("plate"),
                        "height_threshold": 0.1,
                    },
                    "b": {
                        "robot_grasp_cfg": SceneEntityCfg(
                            "robot",
                            body_names=[
                                "ur_robotiq_85_left_finger_tip_link",
                                "ur_robotiq_85_right_finger_tip_link",
                            ],
                        ),
                    },
                },
            },
        )

        drop_plate = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"a": mdp.position_range, "b": mdp.near_one_vector},
                "kwargs": {
                    "a": {
                        "item_cfg": SceneEntityCfg("plate"),
                        "threshold": {
                            "y": (-0.17, None),
                        },
                    },
                    "b": {
                        "item_cfg": SceneEntityCfg("plate"),
                        "vec_ref": torch.tensor([0, 0, 1], dtype=torch.float),
                        "local_vec": torch.tensor([0, 0, 1.0], dtype=torch.float),
                        "lb": -1,
                        "ub": math.sin(math.pi / 180 * 10),
                    },
                },
            },
        )

        place_plate = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"a": mdp.stay_away, "b": mdp.a_in_b()},
                "kwargs": {
                    "a": {
                        "obj_1_cfg": "tcp_transform",
                        "obj_2_cfg": SceneEntityCfg("plate"),
                        "threshold": 0.12,
                    },
                    "b": {
                        "a": SceneEntityCfg("plate"),
                        "b": SceneEntityCfg("right_blanket"),
                        "ratio": 0.2,
                    },
                },
            },
        )

        go_home = ObsTerm(func=mdp.always_true, params={})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()
    images: ImageCfg = ImageCfg()
    camera_info: CameraInfoCfg = CameraInfoCfg()
    depths: DepthCfg = DepthCfg()
    policy_infer = None


@configclass
class OneCameraEventCfg:
    reset_plate_chicken_fork = EventTerm(
        func=mdp.reset_plate_chicken_fork,
        mode="reset",
        params={
            "plate_cfg": SceneEntityCfg("plate"),
            "chicken_cfg": None,  # SceneEntityCfg("chicken"),
            "fork_cfg": SceneEntityCfg("fork"),
            "plate_pose_range": {
                "x": (-0.02, 0.1),
                "y": (-0.14, 0.0),
                # "yaw": (-.1, .1),
            },
        },
    )

    # reset_chicken_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="usd",
    #     params={
    #         "asset_cfg": SceneEntityCfg("chicken"),
    #         "scale_range": {
    #             "x": (0.9, 1.1),
    #             "y": (0.9, 1.1),
    #             "z": (0.9, 1.1),
    #         },
    #     },
    # )

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "pose_range": {
                "z": (-0.04, 0.04),
            },
            "velocity_range": {},
        },
    )

    reset_left_blanket = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("left_blanket"),
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.1),
            },
            "velocity_range": {},
        },
    )

    reset_right_blanket = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_blanket"),
            "pose_range": {
                "x": (-0.02, 0.1),
                "y": (-0.1, 0.02),
            },
            "velocity_range": {},
        },
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
    reset_camera = EventTerm(
        func=mdp.reset_camera_pose,
        mode="reset",
        params={
            "pose_range": {
                # "x": (-0.025, 0.025),
                # "y": (-0.025, 0.025),
                # "z": (-0.04, 0.04),
                # "roll": (-0.05, 0.05),
                # "pitch": (-0.05, 0.05),
                # "yaw": (-0.05, 0.05),
            },
            "asset_cfg": SceneEntityCfg("third_camera"),
        },
    )


@configclass
class UR5CleanPlateMimicEnvCfg(MimicEnvCfg, ManagerBasedRLEnvCfg):
    """Configuration for the UR5 clean plate environment."""

    seed: int = 42
    mimic_config: UR5MimicCfg = UR5MimicCfg()
    observations = OneCameraObs()
    events = OneCameraEventCfg()
    rewards = RewardsCfg()
    terminations = _TimeoutTerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.scene = UR5CleanPlateSceneCfg(num_envs=2, env_spacing=10.0)
        self.mimic_config.default_actions = torch.tensor(
            [0.3090, -0.2433, 0.2945, -0.0892, 0.7072, -0.7005, 0.0349, 0.0000]
        )
        self.decimation = 4
        self.episode_length_s = 120
        # viewer settings
        self.viewer.eye = (-1, -1, 1.8)
        self.viewer.lookat = (-0.0, 0.0, -0.5)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        self.datagen_config.name = "demo_src_clean_plate_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_select_src_per_arm = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = self.seed
        self.sim.physx.enable_ccd = True
        self.actions = UR5ActionsCartesianCfg(dt=self.sim.dt)

        subtask_configs = []
        subtask_configs.append(
            WbcSubTaskConfig(
                object_ref="fork",
                subtask_term_signal="pick_fork",
                first_subtask_start_offset_range=(12, 12),
                subtask_term_offset_range=(6, 6),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                # clip_src_traj_strategy=LastStepClipCfg(last_step_num=40),
                subtask_continous_step=3,
            )
        )

        subtask_configs.append(
            WbcSubTaskConfig(
                object_ref="left_blanket",  # Using fork as cleaning target
                subtask_term_signal="place_fork",
                subtask_term_offset_range=(30, 30),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                # clip_src_traj_strategy=LastStepClipCfg(last_step_num=50),
                subtask_continous_step=5,
            )
        )

        subtask_configs.append(
            WbcSubTaskConfig(
                object_ref="plate",
                subtask_term_signal="pick_plate",
                subtask_term_offset_range=(10, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                # clip_src_traj_strategy=LastStepClipCfg(last_step_num=80),
                subtask_continous_step=3,
            )
        )

        subtask_configs.append(
            WbcSubTaskConfig(
                object_ref="right_blanket",
                subtask_term_signal="drop_plate",
                first_subtask_start_offset_range=(20, 30),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                # clip_src_traj_strategy=LastStepClipCfg(last_step_num=100),
                subtask_continous_step=5,
            )
        )

        subtask_configs.append(
            WbcSubTaskConfig(
                subtask_term_signal="place_plate",
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                # clip_src_traj_strategy=LastStepClipCfg(last_step_num=300),
                subtask_continous_step=3,
                slice_to_last_success_step=True,
            )
        )

        self.subtask_configs["r"] = subtask_configs


# wbc normal
@configclass
class UR5CleanPlateJointWbcMimicCfg(UR5CleanPlateMimicEnvCfg):
    """UR5 clean plate environment."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = UR5JointsActionsCfg()
        self.mimic_config.wbc_solver_cfg = WbcControllerCfg(
            dt=self.sim.dt * self.decimation,
            urdf=f"{PROJECT_ROOT}/source/arxx7_assets/UR5/ur5_isaac_simulation/robot.urdf",
            ee="wrist_3_link",
            active_joint_idx=[0, 1, 2, 3, 4, 5],
            threshold=0.03,
            wbc_type="none",
            p_servo_gain=2,
            v_limit=0.1,
        )
        self.mimic_config.default_actions = torch.tensor(
            [
                -3.5580e00,
                -1.3595e00,
                -2.1516e00,
                -1.0714e00,
                1.7125e00,
                -4.1658e-01,
                0.0,
            ]
        )


# annotate
@configclass
class UR5CleanPlateJointWbcAnnotateMimicCfg(UR5CleanPlateMimicEnvCfg):
    """UR5 clean plate environment."""

    def __post_init__(self):
        super().__post_init__()
        self.mimic_config.wbc_solver_cfg = WbcControllerCfg(
            dt=self.sim.dt * self.decimation,
            urdf=f"{PROJECT_ROOT}/source/arxx7_assets/UR5/ur5_isaac_simulation/robot.urdf",
            ee="wrist_3_link",
            active_joint_idx=[0, 1, 2, 3, 4, 5],
            threshold=0.03,
            wbc_type="none",
            p_servo_gain=1.6,
            v_limit=0.03,
        )


# go home only
@configclass
class UR5CleanPlateJointWbcGoHomeMimicCfg(UR5CleanPlateJointWbcMimicCfg):
    """UR5 clean plate environment."""

    def __post_init__(self):
        super().__post_init__()
        self.subtask_configs["r"].append(
            KeyPointSubTaskConfig(
                key_eef_list=[
                    (0.3090, -0.2433, 0.2945, 0.0892, -0.7072, 0.7005, -0.0349)
                ]
                * 5,
                key_gripper_action_list=[0.0] * 5,
                subtask_term_signal="go_home",
            )
        )


# -----------------------
# Retry


@configclass
class MimicEventCfg:
    reset_plate_during_pick_plate = MimicEventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "asset_cfg": SceneEntityCfg("plate"),
            "pose_range": {
                "x": (-0.02, 0.1),
                "y": (-0.14, 0.0),
            },
            "velocity_range": {},
        },
        trigger_params={
            "subtask_term_signal_ids": [3],
            "prob": 0.005,
        },
        continue_subtask_term_signal="pick_plate",
        eef_name="r",
    )

    reset_plate_fork_during_pick_fork = MimicEventTerm(
        func=mdp.reset_plate_chicken_fork,
        params={
            "plate_cfg": SceneEntityCfg("plate"),
            "chicken_cfg": None,  # SceneEntityCfg("chicken"),
            "fork_cfg": SceneEntityCfg("fork"),
            "plate_pose_range": {
                "x": (-0.02, 0.1),
                "y": (-0.14, 0.0),
            },
        },
        trigger_params={
            "subtask_term_signal_ids": [1],
            "prob": 0.005,
        },
        continue_subtask_term_signal="pick_fork",
        eef_name="r",
    )


# go home + retry
@configclass
class UR5CleanPlateJointWbcGoHomeRetryPlaceMimicCfg(UR5CleanPlateJointWbcMimicCfg):
    """UR5 clean plate environment."""

    def __post_init__(self):
        super().__post_init__()
        self.subtask_configs["r"].append(
            KeyPointSubTaskConfig(
                key_eef_list=[
                    (0.3090, -0.2433, 0.2945, 0.0892, -0.7072, 0.7005, -0.0349)
                ]
                * 5,
                key_gripper_action_list=[0.0] * 5,
                subtask_term_signal="go_home",
            )
        )
        self.mimic_events = MimicEventCfg()


if __name__ == "__main__":
    scene_cfg = UR5CleanPlateMimicEnvCfg()
    print(scene_cfg)
