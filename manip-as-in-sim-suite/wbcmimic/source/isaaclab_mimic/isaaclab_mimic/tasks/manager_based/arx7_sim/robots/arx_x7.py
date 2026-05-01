# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_mimic.utils.path import PROJECT_ROOT

from .. import mdp

##
# Robot definition
##
solver_position_iteration_count = 32
solver_velocity_iteration_count = 0

Arx_X7_CFG_Virtual_Joint = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/X7/usd/x7_with_virtual_joint.usdc",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-3.1, 0.3, 0.05), joint_pos={".*": 0.0}
    ),
    actuators={
        "virtual_joint": ImplicitActuatorCfg(
            joint_names_expr=[f"virtual_joint[12]"],
            effort_limit=1000.0,
            velocity_limit=1,
            stiffness=1e5,
            damping=2e3,
        ),
        "rotate_virtual_joint": ImplicitActuatorCfg(
            joint_names_expr=[f"virtual_joint3"],
            effort_limit=1000.0,
            velocity_limit=0.1,
            stiffness=1e3,
            damping=2e2,
        ),
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint1"],
            effort_limit=3000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=1000,
        ),
        "joint2_3": ImplicitActuatorCfg(
            joint_names_expr=[f"joint2", "joint3"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=10,
            damping=0.2,
        ),
        "joint_arm_1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint4", "joint13"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=300,
            damping=5,
        ),
        "joint_arm_2_4": ImplicitActuatorCfg(
            joint_names_expr=[
                f"joint5",
                "joint6",
                "joint7",
                "joint14",
                "joint15",
                "joint16",
            ],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=200,
            damping=5,
        ),
        "joint_arm_5_6": ImplicitActuatorCfg(
            joint_names_expr=[f"joint8", "joint9", "joint17", "joint18"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=100,
            damping=1,
        ),
        "joint_arm_7": ImplicitActuatorCfg(
            joint_names_expr=[f"joint10", "joint19"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=80,
            damping=1,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[f"joint11", "joint12", "joint20", "joint21"],
            effort_limit=50.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=5000,
            damping=200,
            # stiffness=0,
            # damping=0.2
        ),
    },
)

# Deprecated
Arx_X7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/X7/usd/final/X7.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-3.1, 0.3, 0.05), joint_pos={".*": 0.0}
    ),
    actuators={
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint1"],
            effort_limit=3000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=1000,
        ),
        "joint2_3": ImplicitActuatorCfg(
            joint_names_expr=[f"joint2", "joint3"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=10,
            damping=0.2,
        ),
        "joint_arm_1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint4", "joint13"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=300,
            damping=5,
        ),
        "joint_arm_2_4": ImplicitActuatorCfg(
            joint_names_expr=[
                f"joint5",
                "joint6",
                "joint7",
                "joint14",
                "joint15",
                "joint16",
            ],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=200,
            damping=5,
        ),
        "joint_arm_5_6": ImplicitActuatorCfg(
            joint_names_expr=[f"joint8", "joint9", "joint17", "joint18"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=100,
            damping=1,
        ),
        "joint_arm_7": ImplicitActuatorCfg(
            joint_names_expr=[f"joint10", "joint19"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=80,
            damping=1,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[f"joint11", "joint12", "joint20", "joint21"],
            effort_limit=50.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=5000,
            damping=200,
            # stiffness=0,
            # damping=0.2
        ),
    },
)

Arx_X7_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf",
        activate_contact_sensors=True,
        usd_dir=f"{PROJECT_ROOT}/source/arxx7_assets/X7/debug",
        usd_file_name="X7.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=20.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None, damping=None
            )
        ),
        collider_type="convex_decomposition",
        replace_cylinders_with_capsules=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-3.1, 0.3, 0.05), joint_pos={".*": 0.0}
    ),
    actuators={
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint1"],
            effort_limit=20000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=1000,
        ),
        "joint2_3": ImplicitActuatorCfg(
            joint_names_expr=[f"joint2", "joint3"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=10,
            damping=0.2,
        ),
        "joint_arm_1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint4", "joint13"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=300,
            damping=5,
        ),
        "joint_arm_2_4": ImplicitActuatorCfg(
            joint_names_expr=[
                f"joint5",
                "joint6",
                "joint7",
                "joint14",
                "joint15",
                "joint16",
            ],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=200,
            damping=5,
        ),
        "joint_arm_5_6": ImplicitActuatorCfg(
            joint_names_expr=[f"joint8", "joint9", "joint17", "joint18"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=100,
            damping=1,
        ),
        "joint_arm_7": ImplicitActuatorCfg(
            joint_names_expr=[f"joint10", "joint19"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=80,
            damping=1,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[f"joint11", "joint12", "joint20", "joint21"],
            effort_limit=50.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=10,
            damping=1,
        ),
    },
)

Arx_X7_URDF_CFG_hard = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf",
        activate_contact_sensors=True,
        usd_dir=f"{PROJECT_ROOT}/source/arxx7_assets/X7/usd",
        usd_file_name="X7.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=20.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        fix_base=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None, damping=None
            )
        ),
        collider_type="convex_decomposition",
        replace_cylinders_with_capsules=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-3.1, 0.3, 0.05),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint1"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=1000,
        ),
        "joint2_3": ImplicitActuatorCfg(
            joint_names_expr=[f"joint2", "joint3"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=1000,
            damping=20,
        ),
        "joint_arm_1": ImplicitActuatorCfg(
            joint_names_expr=[f"joint4", "joint13"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=500,
        ),
        "joint_arm_2_4": ImplicitActuatorCfg(
            joint_names_expr=[
                f"joint5",
                "joint6",
                "joint7",
                "joint14",
                "joint15",
                "joint16",
            ],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=500,
        ),
        "joint_arm_5_6": ImplicitActuatorCfg(
            joint_names_expr=[f"joint8", "joint9", "joint17", "joint18"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=100,
        ),
        "joint_arm_7": ImplicitActuatorCfg(
            joint_names_expr=[f"joint10", "joint19"],
            effort_limit=1000.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=50000,
            damping=100,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[f"joint11", "joint12", "joint20", "joint21"],
            effort_limit=400.0,
            # effort_limit=500.0,
            velocity_limit=0.1,
            stiffness=400,
            damping=10,
        ),
    },
)


@configclass
class X7JointActionsCfg:
    """Action specifications for the MDP."""

    # velocity should be at first to match with wbc
    velocity = mdp.VelocityActionCfg(asset_name="robot")
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"joint{i}" for i in range(1, 22)],
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class X7ActionsCartesianCfg:
    """Action specifications for the MDP."""

    dt = 1 / 120
    cartesian = mdp.CartesianPositionActionCfg(
        asset_name="robot",
        joint_names=[f"joint{i}" for i in range(1, 22)],
        preserve_order=True,
        use_in_lp_filter=True,
        use_ik_solve_lift=False,
    )
    velocity = mdp.VelocityActionCfg(asset_name="robot")

    def __post_init__(self) -> None:
        self.cartesian.dt = self.dt
        del self.dt


DEFAULT_X7_JOINT_MOBILE_ACTIONS = torch.tensor([0.0] * 24)

DEFAULT_X7_CARTESIAN_ACTIONS = torch.tensor(
    [
        0.35,
        0.0,
        0.0,
        0.16,
        0.50,
        0.35 - 0.03,
        1,
        0,
        0,
        0,
        0.16,
        -0.50,
        0.35 - 0.03,
        1,
        0,
        0,
        0,
        0.044,
        0.044,
        0,
        0,
        0,
    ]
)


@configclass
class X7ObservationsCfg:
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
                    joint_names=[f"joint{i}" for i in range(1, 22)],
                    preserve_order=True,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[f"joint{i}" for i in range(1, 22)],
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


def ee_frame_pos(
    env: ManagerBasedRLEnv,
    x7_entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left=True,
) -> torch.Tensor:
    robot = env.scene[x7_entity_cfg.name]

    x7_entity_cfg.body_names = ["link10"] if left else ["link19"]
    x7_entity_cfg.resolve(env.scene)
    ee_id = x7_entity_cfg.body_ids[0]
    root_entity_cfg = SceneEntityCfg("robot", body_names=["base_link"])
    root_entity_cfg.resolve(env.scene)
    root_id = root_entity_cfg.body_ids[0]

    ee_pose_w = robot.data.body_state_w[:, ee_id, 0:7]
    root_pose_w = robot.data.body_state_w[:, root_id, 0:7]
    ee_pose_b = torch.cat(
        subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        ),
        dim=1,
    )
    return ee_pose_b[:, :3]


def ee_frame_quat(
    env: ManagerBasedRLEnv,
    x7_entity_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left=True,
) -> torch.Tensor:
    robot = env.scene[x7_entity_cfg.name]

    x7_entity_cfg.body_names = ["link10"] if left else ["link19"]
    x7_entity_cfg.resolve(env.scene)
    ee_id = x7_entity_cfg.body_ids[0]
    root_entity_cfg = SceneEntityCfg("robot", body_names=["base_link"])
    root_entity_cfg.resolve(env.scene)
    root_id = root_entity_cfg.body_ids[0]

    ee_pose_w = robot.data.body_state_w[:, ee_id, 0:7]
    root_pose_w = robot.data.body_state_w[:, root_id, 0:7]
    ee_pose_b = torch.cat(
        subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        ),
        dim=1,
    )
    return ee_pose_b[:, 3:]


@configclass
class X7ObservationsMimicCfg:
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
                    joint_names=[f"joint{i}" for i in range(1, 22)],
                    preserve_order=True,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[f"joint{i}" for i in range(1, 22)],
                    preserve_order=True,
                ),
            },
        )
        base_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["virtual_joint1", "virtual_joint2", "virtual_joint3"],
                    preserve_order=True,
                ),
            },
        )

        lift_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["joint1"],
                    preserve_order=True,
                ),
            },
        )

        left_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[f"joint{i}" for i in range(4, 11)],
                    preserve_order=True,
                ),
            },
        )

        right_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[f"joint{i}" for i in range(13, 20)],
                    preserve_order=True,
                ),
            },
        )

        left_arm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[f"joint{i}" for i in range(4, 11)],
                    preserve_order=True,
                ),
            },
        )
        right_arm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[f"joint{i}" for i in range(13, 20)],
                    preserve_order=True,
                ),
            },
        )

        # 0 for open
        left_normalized_gripper_pos = ObsTerm(
            func=mdp.one_hot_gripper_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["joint11"],
                    preserve_order=True,
                ),
                "threshold": 0.04,
                "reverse": True,
            },
        )

        right_normalized_gripper_pos = ObsTerm(
            func=mdp.one_hot_gripper_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["joint20"],
                    preserve_order=True,
                ),
                "threshold": 0.04,
                "reverse": True,
            },
        )

        eef_pos_r = ObsTerm(
            func=ee_frame_pos,
            params={
                "x7_entity_cfg": SceneEntityCfg("robot"),
                "left": False,
            },
        )

        eef_quat_r = ObsTerm(
            func=ee_frame_quat,
            params={
                "x7_entity_cfg": SceneEntityCfg("robot"),
                "left": False,
            },
        )

        eef_pos_l = ObsTerm(
            func=ee_frame_pos,
            params={
                "x7_entity_cfg": SceneEntityCfg("robot"),
                "left": True,
            },
        )

        eef_quat_l = ObsTerm(
            func=ee_frame_quat,
            params={
                "x7_entity_cfg": SceneEntityCfg("robot"),
                "left": True,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class DualArmMimicgenConfig:

    lift: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    left: bool = False
    default_actions: torch.Tensor = DEFAULT_X7_CARTESIAN_ACTIONS
    warmup_steps = 40
    wbc_solver_cfg = None
    episode_end_when_wbc_failed = False
