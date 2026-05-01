# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_mimic.utils.path import PROJECT_ROOT

from .. import mdp

##
# Robot definition
##
solver_position_iteration_count = 32

# UR5_CFG = ArticulationCfg(
#     # spawn=sim_utils.UrdfFileCfg(
#     #     asset_path=f"{PROJECT_ROOT}/source/arxx7_assets/UR5/ur5_isaac_simulation/robot.urdf",
#     #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
#     #         rigid_body_enabled=True,
#     #         max_linear_velocity=20.0,
#     #         max_angular_velocity=20.0,
#     #         max_depenetration_velocity=20.0,
#     #         enable_gyroscopic_forces=True,
#     #     ),
#     #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#     #         enabled_self_collisions=False,
#     #         solver_position_iteration_count=solver_position_iteration_count,
#     #         solver_velocity_iteration_count=0,
#     #         sleep_threshold=0.005,
#     #         stabilization_threshold=0.001,
#     #     ),
#     #     fix_base=True,
#     #     joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#     #         gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
#     #             stiffness=None, damping=None
#     #         )
#     #     ),
#     #     collider_type="convex_decomposition",
#     # ),
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/UR5/usd/ur5_2.usdc",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=20.0,
#             max_angular_velocity=20.0,
#             max_depenetration_velocity=20.0,
#             enable_gyroscopic_forces=True,
#             disable_gravity=True,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=solver_position_iteration_count,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#         ),
#         joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="acceleration")
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(-3.1, 0.3, 0.05),
#         joint_pos={
#             "elbow_joint": -1.50796447,
#             "gripper_joint": 0.0,
#             "ur_robotiq_85_.*": 0.0,
#             "shoulder_lift_joint": -2.00677957,
#             "shoulder_pan_joint": -3.02692452,
#             "wrist_1_joint": -1.12242124,
#             "wrist_2_joint": 1.59191481,
#             "wrist_3_joint": -0.055676,
#             # "elbow_joint": -1.5239,
#             # "gripper_joint": 0.0,
#             # "ur_robotiq_85_.*": 0.0,
#             # "shoulder_lift_joint": -1.677,
#             # "shoulder_pan_joint": -2.5451,
#             # "wrist_1_joint": 3.1749,
#             # "wrist_2_joint": -0.634,
#             # "wrist_3_joint": -1.5618,
#         },
#     ),
#     actuators={
#         "arm": ImplicitActuatorCfg(
#             joint_names_expr=[f"elbow_joint",
#                                "shoulder_lift_joint",
#                                "shoulder_pan_joint", "wrist_._joint"],
#             effort_limit=20000.0,
#             # effort_limit=500.0,
#             velocity_limit=100.0,
#             stiffness=1e15,
#             damping=5e2,
#         ),
#         "gripper": ImplicitActuatorCfg(
#             joint_names_expr=["gripper_joint",
#             "ur_robotiq_85_.*"],
#             effort_limit=20.0,
#             # effort_limit=500.0,
#             velocity_limit=100.0,
#             stiffness=1e15,
#             damping=1e14,
#         )
#     }
# )

UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/UR5/usd/ur5_simplify_on_urdf_gripper.usdc",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=20.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="acceleration"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-3.1, 0.3, 0.05),
        joint_pos={
            "elbow_joint": -1.50796447,
            "gripper_joint": 0.0,
            "ur_robotiq_85_.*": 0.0,
            "shoulder_lift_joint": -2.00677957,
            "shoulder_pan_joint": -3.02692452,
            "wrist_1_joint": -1.12242124,
            "wrist_2_joint": 1.59191481,
            "wrist_3_joint": -0.055676,
            # "elbow_joint": -1.5239,
            # "gripper_joint": 0.0,
            # "ur_robotiq_85_.*": 0.0,
            # "shoulder_lift_joint": -1.677,
            # "shoulder_pan_joint": -2.5451,
            # "wrist_1_joint": 3.1749,
            # "wrist_2_joint": -0.634,
            # "wrist_3_joint": -1.5618,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                f"elbow_joint",
                "shoulder_lift_joint",
                "shoulder_pan_joint",
                "wrist_._joint",
            ],
            effort_limit=None,
            # effort_limit=500.0,
            velocity_limit=None,
            stiffness=1e15,
            damping=5e2,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "gripper_joint",
                # 'ur_robotiq_85_left_inner_knuckle_joint',
                # 'ur_robotiq_85_right_inner_knuckle_joint',
                "ur_robotiq_85_right_knuckle_joint",
                "ur_robotiq_85_left_finger_tip_joint",
                "ur_robotiq_85_right_finger_tip_joint",
            ],
            effort_limit=20.0,
            # effort_limit=500.0,
            velocity_limit=100.0,
            stiffness=1e15,
            damping=1e14,
        ),
        "knuckle_joint": ImplicitActuatorCfg(
            joint_names_expr=[
                "ur_robotiq_85_left_inner_knuckle_joint",
                "ur_robotiq_85_right_inner_knuckle_joint",
            ],
            effort_limit=20.0,
            # effort_limit=500.0,
            velocity_limit=100.0,
            stiffness=0,
            damping=0,
        ),
    },
)


@configclass
class UR5JointsActionsCfg:
    """Action specifications for the MDP."""

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        preserve_order=True,
        use_default_offset=False,
    )

    gripper_action = mdp.GripperControlCfg(
        asset_name="robot",
        joint_names=[
            "gripper_joint",
            "ur_robotiq_85_left_inner_knuckle_joint",
            "ur_robotiq_85_right_inner_knuckle_joint",
            "ur_robotiq_85_right_knuckle_joint",
            "ur_robotiq_85_left_finger_tip_joint",
            "ur_robotiq_85_right_finger_tip_joint",
        ],
    )


@configclass
class UR5ActionsCartesianCfg:
    """Action specifications for the MDP."""

    dt = 1 / 120
    cartesian = mdp.CartesianPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "gripper_joint",
            "ur_robotiq_85_left_inner_knuckle_joint",
            "ur_robotiq_85_right_inner_knuckle_joint",
            "ur_robotiq_85_right_knuckle_joint",
            "ur_robotiq_85_left_finger_tip_joint",
            "ur_robotiq_85_right_finger_tip_joint",
        ],
        preserve_order=True,
        use_in_lp_filter=True,
    )

    def __post_init__(self) -> None:
        self.cartesian.dt = self.dt
        del self.dt


@configclass
class UR5ActionsRelativeCartesianCfg(UR5ActionsCartesianCfg):
    """Action specifications for the MDP."""

    cartesian = mdp.RelativeCartesianPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "gripper_joint",
            "ur_robotiq_85_left_inner_knuckle_joint",
            "ur_robotiq_85_right_inner_knuckle_joint",
            "ur_robotiq_85_right_knuckle_joint",
            "ur_robotiq_85_left_finger_tip_joint",
            "ur_robotiq_85_right_finger_tip_joint",
        ],
        preserve_order=True,
        use_in_lp_filter=True,
    )


DEFAULT_UR5_JOINT_ACTIONS = torch.tensor(
    [-3.02692452, -2.00677957, -1.50796447, -1.12242124, 1.59191481, -0.055676, 0.0]
)

DEFAULT_UR5_CARTESIAN_ACTIONS = torch.tensor(
    [0.6534, -0.0329, 0.2565, 0.0346, -0.7644, 0.6436, -0.0181, 0.0]
)


@configclass
class UR5ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                        "gripper_joint",
                    ],
                    preserve_order=True,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                        "gripper_joint",
                    ],
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5ObservationsMimicCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                    ],
                    preserve_order=True,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                    ],
                    preserve_order=True,
                ),
            },
        )

        eef_pos = ObsTerm(
            func=mdp.ee_frame_pos,
            params={
                "entity_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]),
            },
        )

        eef_quat = ObsTerm(
            func=mdp.ee_frame_orn,
            params={
                "entity_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]),
            },
        )

        last_action = ObsTerm(
            func=mdp.last_action,
        )

        abs_gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["gripper_joint"],
                    preserve_order=True,
                ),
            },
        )

        normalized_gripper_pos = ObsTerm(
            func=mdp.one_hot_gripper_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["gripper_joint"],
                    preserve_order=True,
                ),
                "threshold": 0.1,
                "reverse": False,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5MimicCfg:
    """Mimic specifications for the MDP."""

    # observation terms (order preserved)
    default_actions = DEFAULT_UR5_CARTESIAN_ACTIONS
    warmup_steps = 20
    wbc_solver_cfg = None
    rl_solver_cfg = None
    episode_end_when_wbc_failed = False
    """whether to end the episode when wbc failed"""
