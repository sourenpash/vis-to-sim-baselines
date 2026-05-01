# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_mimic.envs import KeyPointSubTaskConfig, WbcSubTaskConfig
from isaaclab_mimic.envs.clip_src_traj_strategy import ClipFarPosesCfg
from isaaclab_mimic.utils.path import PROJECT_ROOT
from isaaclab_mimic.utils.rigid_body import RigidObjectInitDataAuto
from isaaclab_mimic.utils.robots.wbc_controller_dual import DualArmWbcControllerCfg

from . import mdp
from .robots import (
    DEFAULT_X7_JOINT_MOBILE_ACTIONS,
    Arx_X7_CFG,
    Arx_X7_CFG_Virtual_Joint,
    Arx_X7_URDF_CFG,
    Arx_X7_URDF_CFG_hard,
    DualArmMimicgenConfig,
    X7ActionsCartesianCfg,
    X7JointActionsCfg,
    X7ObservationsMimicCfg,
)
from .room import EventsCfg, WashTableSceneCfg


@configclass
class _X7AliveRewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class X7PickToothpasteIntoCupAndPushSceneCfg(WashTableSceneCfg):
    """Configuration for the X7 pick toothpaste into cup and push environment."""

    robot = Arx_X7_CFG_Virtual_Joint.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(3.13358, -1.13374, 0.18),
            rot=(0.70710678, 0.0, 0.0, -0.70710678),
            joint_pos={
                "virtual.*": 0.0,
                "joint1": 0.55,
                "joint[23]": 0.0,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": 0.0,
                "joint8": 0.0,
                "joint9": 0.0,
                "joint10": 0.0,
                "joint11": 0.044,
                "joint12": 0.044,
                "joint13": 0.0,
                "joint14": 0.0,
                "joint15": 0.0,
                "joint16": 0.0,
                "joint17": 0.0,
                "joint18": 0.0,
                "joint19": 0.0,
                "joint20": 0.044,
                "joint21": 0.044,
            },
        ),
    )

    robot_left_arm_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link10/left_arm_camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.001, 20),
            focal_length=1.020186901041297,
            horizontal_aperture=2.015479769898128,
            vertical_aperture=1.1350198537598315,
            horizontal_aperture_offset=0.007303626830546249,
            vertical_aperture_offset=0.0014308687570375137,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.06503521503112178, 0.031719842184532436, 0.04267751397724456),
            rot=(0.95188393, 0.293583, 0.07652446, 0.04324372),
            convention="ros",
        ),
    )

    robot_right_arm_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link19/right_arm_camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.001, 20),
            focal_length=1.020186901041297,
            horizontal_aperture=1.9751669369101887,
            vertical_aperture=1.1079447453791091,
            horizontal_aperture_offset=-0.005917874284638102,
            vertical_aperture_offset=-0.012909419875904372,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.00850199334704118, -0.02876797096954271, 0.08661617266547979),
            rot=(0.95291347, -0.09096244, 0.01259262, 0.28900379),
            convention="ros",
        ),
    )

    robot_head_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link1/head_camera",
        update_period=0,
        height=480,
        width=848,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.001, 20),
            focal_length=1.88,
            horizontal_aperture=3.675232,
            vertical_aperture=2.08032,
            horizontal_aperture_offset=-0.019998,
            vertical_aperture_offset=-0.011912,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # pos=(0.333019633186, -0.27608455233828244, 0.12017731463056525), # y 轴增加，向左
            # pos=(0.333019633186, -0.2, 0.14), # y 轴增加，向左
            # rot=(0.25056, -0.66308, 0.66142, -0.24172),
            pos=(0.25869355, -0.02536336, 0.2195015),
            rot=(0.25494778, -0.66399991, 0.66351055, -0.23207652),
            convention="ros",
        ),
    )

    record_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/record_camera",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.001, 20),
            # focal_length=1.020186901041297,
            # horizontal_aperture=1.9751669369101887,
            # vertical_aperture=1.1079447453791091,
            # horizontal_aperture_offset=-0.005917874284638102,
            # vertical_aperture_offset=-0.012909419875904372,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.87238, -1.96936, 1.17957),
            rot=(-0.46745, -0.38959, 0.50, 0.60),
        ),
    )

    def __post_init__(self) -> None:

        self.cup = RigidObjectCfg(
            class_type=RigidObjectInitDataAuto,
            prim_path="{ENV_REGEX_NS}/cup",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(2.85, -2.46, 0.95), rot=(1, 0, 0, 0)
            ),
            spawn=UsdFileCfg(
                usd_path=f"{str(Path(self.usd_path).parent.parent)}/Cup001/Cup001.usda",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.toothpaste = RigidObjectCfg(
            class_type=RigidObjectInitDataAuto,
            prim_path="{ENV_REGEX_NS}/toothpaste",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(3, -2.27, 0.9415),
                # rot=(0.5, 0.5, -0.5, -0.5),
                rot=(0.707, 0.0, 0.0, 0.707),
            ),
            spawn=UsdFileCfg(
                usd_path=f"{str(Path(self.usd_path).parent.parent)}/Toothpaste001/Toothpaste001.usda",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.link11_cf = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link11",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/toothpaste/*"],
        )

        self.link12_cf = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link12",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/toothpaste/*"],
        )
        self.link20_cf = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link20",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/toothpaste/*"],
        )

        self.link21_cf = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link21",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/toothpaste/*"],
        )
        del self.usd_path


@configclass
class X7PickToothpasteIntoCupAndPushEventsCfg(EventsCfg):

    reset_cup = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "pose_range": {
                "x": (-0.2, 0.0),
                "y": (-0.0, 0.15),
                "yaw": (1.5, 1.5),
            },
            "velocity_range": {},
        },
    )

    reset_toothpaste = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("toothpaste"),
            "pose_range": {
                "x": (-0.0, 0.3),
                "y": (-0.1, 0.0),
                # "pitch": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )

    reset_camera = EventTerm(
        func=mdp.reset_camera_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.025, 0.025),
                "y": (-0.025, 0.025),
                "z": (-0.04, 0.04),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "asset_cfg": SceneEntityCfg("robot_head_camera"),
        },
    )


@configclass
class X7PickToothpasteIntoCupAndPushTermCfg:
    timeout = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # success = DoneTerm(
    #     func=mdp.and_func,
    #     time_out=False,
    #     params={
    #         "funcs": {"a": mdp.position_range, "b": mdp.a_in_b()},
    #         "kwargs": {
    #             "a": {
    #                 "item_cfg": SceneEntityCfg("cup"),
    #                 "threshold": {'y': (None, -2.7),}
    #             },
    #             "b": {
    #                 "a":  SceneEntityCfg("toothpaste"),
    #                 "b":  SceneEntityCfg("cup"),
    #                 "ratio": 0.35,
    #             }
    #         }
    #     },
    # )

    # TODO
    early_stop = DoneTerm(
        func=mdp.always_false,
        time_out=False,
        params={},
    )

    success = DoneTerm(
        func=mdp.success_all_task_sequentially,
        time_out=False,
        params={},
    )


@configclass
class X7PickToothpasteIntoCupAndPushObservationsCfg(X7ObservationsMimicCfg):

    @configclass
    class ImageCfg(ObsGroup):
        """Observations for policy group."""

        # robot_left_arm_camera = ObsTerm(
        #     func=mdp.get_camera_image,
        #     params={
        #     "camera_entity": SceneEntityCfg("robot_left_arm_camera"),
        #     },
        # )
        # robot_right_arm_camera = ObsTerm(
        #     func=mdp.get_camera_image,
        #     params={
        #         "camera_entity": SceneEntityCfg("robot_right_arm_camera"),
        #     }
        # )
        camera_0 = ObsTerm(
            func=mdp.get_camera_image,
            params={
                "camera_entity": SceneEntityCfg("robot_head_camera"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraInfoCfg(ObsGroup):
        # robot_left_arm_camera_intr = ObsTerm(
        #     func=mdp.get_camera_intr,
        #     params={
        #         "camera_entity": SceneEntityCfg("robot_left_arm_camera"),
        #     }
        # )

        # robot_left_arm_camera_extr = ObsTerm(
        #     func=mdp.get_camera_extr,
        #     params={
        #         "camera_entity": SceneEntityCfg("robot_left_arm_camera"),
        #     }
        # )

        # robot_right_arm_camera_intr = ObsTerm(
        #     func=mdp.get_camera_intr,
        #     params={
        #         "camera_entity": SceneEntityCfg("robot_right_arm_camera"),
        #     }
        # )

        # robot_right_arm_camera_extr = ObsTerm(
        #     func=mdp.get_camera_extr,
        #     params={
        #         "camera_entity": SceneEntityCfg("robot_right_arm_camera")
        #     }
        # )

        robot_head_camera_intr = ObsTerm(
            func=mdp.get_camera_intr,
            params={
                "camera_entity": SceneEntityCfg("robot_head_camera"),
            },
        )

        robot_head_camera_extr = ObsTerm(
            func=mdp.get_camera_extr,
            params={"camera_entity": SceneEntityCfg("robot_head_camera")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DepthCfg(ObsGroup):
        # robot_left_arm_camera = ObsTerm(
        #     func=mdp.get_camera_depth,
        #     params={
        #         "camera_entity": SceneEntityCfg("robot_left_arm_camera"),
        #     },
        # )
        # robot_right_arm_camera = ObsTerm(
        #     func=mdp.get_camera_depth,
        #     params={
        #         "camera_entity": SceneEntityCfg("robot_right_arm_camera"),
        #     },
        # )
        camera_0 = ObsTerm(
            func=mdp.get_camera_depth,
            params={
                "camera_entity": SceneEntityCfg("robot_head_camera"),
                "resize": True,
                "resize_shape": (240, 320),  # Resize to 240x320
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        # approach_toothpaste = ObsTerm(
        #     func=mdp.approach_success,
        #     params={
        #         "item_cfg": SceneEntityCfg("toothpaste"),
        #         "robot_cfg": SceneEntityCfg("robot", body_names=["link20", "link21"]),
        #         "threshold": 0.1,
        #     },
        # )

        # pick_toothpaste = ObsTerm(
        #     func=mdp.lift_success,
        #     params={
        #         "item_cfg": SceneEntityCfg("toothpaste"),
        #         "robot_grasp_cfg": SceneEntityCfg("robot", body_names=["link11", "link12"]),
        #         "height_threshold": 0.02,
        #     },
        # )

        pick_toothpaste = ObsTerm(
            func=mdp.lift_success_simple,
            params={
                "item_cfg": SceneEntityCfg("toothpaste"),
                # "robot_grasp_cfg": SceneEntityCfg("robot", body_names=["link11", "link12"]),
                "height_threshold": 0.03,
            },
        )

        place_toothpaste_in_cup = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {"a": mdp.a_in_b(), "b": mdp.robot_joint_pos_success},
                "kwargs": {
                    "a": {
                        "a": SceneEntityCfg("toothpaste"),
                        "b": SceneEntityCfg("cup"),
                        "ratio": 0.35,
                    },
                    "b": {
                        "robot_entity": SceneEntityCfg(
                            "robot", joint_names=["joint11"]
                        ),
                        "threshold": (0.04, 0.05),
                    },
                },
            },
        )

        toothpaste_in_cup_and_push = ObsTerm(
            func=mdp.and_func,
            params={
                "funcs": {
                    "a": mdp.position_range,
                    "b": mdp.a_in_b(),
                    "c": mdp.near_one_vector,
                },
                "kwargs": {
                    "a": {
                        "item_cfg": SceneEntityCfg("cup"),
                        "threshold": {
                            "y": (None, -2.5),
                        },
                    },
                    "b": {
                        "a": SceneEntityCfg("toothpaste"),
                        "b": SceneEntityCfg("cup"),
                        "ratio": 0.35,
                    },
                    "c": {
                        "item_cfg": SceneEntityCfg("cup"),
                        "vec_ref": torch.tensor([0, 0, 1.0], dtype=torch.float),
                        "local_vec": torch.tensor([0, 0, 1.0], dtype=torch.float),
                        "lb": 0.9,
                        "ub": 1.1,
                    },
                },
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()
    images: ImageCfg = ImageCfg()
    camera_info: CameraInfoCfg = CameraInfoCfg()
    depths: DepthCfg = DepthCfg()


@configclass
class X7PickToothpasteIntoCupAndPushMimicCfg(MimicEnvCfg, ManagerBasedRLEnvCfg):
    seed: int = 42
    usd_path: str = (
        f"{PROJECT_ROOT}/source/arxx7_assets/Collected_Room_empty_all_table/Room_empty_table.usdc"
    )

    mimic_config: DualArmMimicgenConfig = DualArmMimicgenConfig(left=False, lift=0.35)

    observations = X7PickToothpasteIntoCupAndPushObservationsCfg()
    # actions = X7ActionsCartesianCfg()
    events = X7PickToothpasteIntoCupAndPushEventsCfg()
    rewards = _X7AliveRewardsCfg()
    terminations = X7PickToothpasteIntoCupAndPushTermCfg()

    def __post_init__(self) -> None:
        self.scene = X7PickToothpasteIntoCupAndPushSceneCfg(
            num_envs=1, env_spacing=40.0, usd_path=self.usd_path
        )
        self.decimation = 4
        self.episode_length_s = 120
        # viewer settings
        self.viewer.eye = (1.96358, -0.72374, 2.2)
        self.viewer.lookat = (2.96358, -2.52374, 0.50724)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.datagen_config.name = "demo_src_approach_toothpaste_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = self.seed
        self.actions = X7ActionsCartesianCfg(self.sim.dt)

        subtask_configs_l = []
        subtask_configs_l.append(
            WbcSubTaskConfig(
                object_ref="toothpaste",
                subtask_term_signal="pick_toothpaste",
                subtask_term_offset_range=(14, 14),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                clip_src_traj_strategy=ClipFarPosesCfg(distance_threshold=0.1),
                wbc_max_step_start=500,
                wbc_max_step=3,
                wbc_max_step_last=500,
            )
        )
        subtask_configs_l.append(
            WbcSubTaskConfig(
                object_ref="cup",
                subtask_term_signal="place_toothpaste_in_cup",
                subtask_term_offset_range=(9, 9),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                # slice_to_last_success_step=True,
                clip_src_traj_strategy=ClipFarPosesCfg(distance_threshold=0.1),
                wbc_max_step_start=200,
                wbc_max_step=3,
                wbc_max_step_last=200,
            )
        )
        subtask_configs_l.append(
            WbcSubTaskConfig(
                subtask_term_signal="toothpaste_in_cup_and_push",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                clip_src_traj_strategy=ClipFarPosesCfg(),
                wbc_max_step_start=200,
                wbc_max_step=3,
                slice_to_last_success_step=True,
                wbc_max_step_last=200,
                subtask_continous_step=2,
            )
        )
        subtask_configs_r = []
        subtask_configs_r.append(
            WbcSubTaskConfig(
                subtask_term_signal="pick_toothpaste",
                subtask_term_offset_range=(14, 14),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                clip_src_traj_strategy=ClipFarPosesCfg(distance_threshold=0.1),
                wbc_max_step_start=500,
                wbc_max_step=3,
                wbc_max_step_last=500,
            )
        )
        subtask_configs_r.append(
            WbcSubTaskConfig(
                subtask_term_signal="place_toothpaste_in_cup",
                subtask_term_offset_range=(9, 9),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                # slice_to_last_success_step=True,
                clip_src_traj_strategy=ClipFarPosesCfg(distance_threshold=0.1),
                wbc_max_step_start=200,
                wbc_max_step=3,
                wbc_max_step_last=200,
            )
        )
        subtask_configs_r.append(
            WbcSubTaskConfig(
                object_ref="cup",
                subtask_term_signal="toothpaste_in_cup_and_push",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=30,
                num_fixed_steps=5,
                apply_noise_during_interpolation=False,
                clip_src_traj_strategy=ClipFarPosesCfg(),
                slice_to_last_success_step=True,
                wbc_max_step_start=200,
                wbc_max_step=3,
                wbc_max_step_last=200,
                subtask_continous_step=2,
            )
        )

        self.subtask_configs["l"] = subtask_configs_l
        self.subtask_configs["r"] = subtask_configs_r
        # self.task_constraint_configs = [
        #     SubTaskConstraintConfig(
        #         eef_subtask_constraint_tuple=[("l", 1), ("r", 0)],
        #         constraint_type=SubTaskConstraintType.COORDINATION,
        #     )
        # ]


@configclass
class X7PickToothpasteIntoCupAndPushJointWbcMimicCfgAnnotated(
    X7PickToothpasteIntoCupAndPushMimicCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.mimic_config.wbc_solver_cfg = DualArmWbcControllerCfg(
            dt=self.sim.dt * self.decimation,
            urdf=f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7_2.urdf",
            ees=["link10", "link19"],
            active_joint_idx=[
                0,
                1,
                2,
                3,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
            ],
            threshold=0.05,
        )


@configclass
class X7PickToothpasteIntoCupAndPushJointWbcMimicCfg(
    X7PickToothpasteIntoCupAndPushMimicCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.mimic_config.wbc_solver_cfg = DualArmWbcControllerCfg(
            dt=self.sim.dt * self.decimation,
            urdf=f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7_2.urdf",
            ees=["link10", "link19"],
            active_joint_idx=[
                0,
                1,
                2,
                3,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
            ],
            threshold=0.05,
        )
        # self.mimic_config.default_actions = torch.tensor([
        #     0, 0, 0,
        #     0.55, 0, 0,
        #     0.17, -1.39, -1.4, 0.2, 0.069, -1.6, 1.6,
        #     0.044, 0.044,
        #     -0.17, 1.39, 1.4, -0.2, -0.069, 1.6, -1.6,
        #     0, 0,
        # ])
        self.mimic_config.default_actions = torch.tensor(
            [
                0,
                0,
                0,
                0.55,
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.044,
                0.044,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.044,
                0.044,
            ]
        )
        self.actions = X7JointActionsCfg()


@configclass
class X7PickToothpasteIntoCupAndPushEventsRotateCfg(
    X7PickToothpasteIntoCupAndPushEventsCfg
):
    reset_cup = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cup"),
            "pose_range": {
                "x": (-0.2, 0.0),
                "y": (-0.0, 0.15),
                "yaw": (0.0, 1.5),
            },
            "velocity_range": {},
        },
    )

    reset_toothpaste = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("toothpaste"),
            "pose_range": {
                "x": (-0.0, 0.3),
                "y": (-0.1, 0.0),
                "pitch": (-0.2, 0.2),
            },
            "velocity_range": {},
        },
    )


@configclass
class X7PickToothpasteIntoCupAndPushJointRotateWbcMimicCfg(
    X7PickToothpasteIntoCupAndPushJointWbcMimicCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.events = X7PickToothpasteIntoCupAndPushEventsRotateCfg()
