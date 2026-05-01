# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import random
import time

import isaaclab.utils.math as math_utils
import numpy as np
import pink
import pinocchio as pin
import qpsolvers
import torch
from isaaclab_mimic.utils.path import PROJECT_ROOT
from isaaclab_mimic.utils.robot_ik import UR5PinkIK_Curobo, UR5PinkIK_Pink
from loguru import logger
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask


class CartesianHelper_Pink:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: list,
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 0.5,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        env_num: int = 1,
    ):
        self.ik_policy_dummy = UR5PinkIK_Pink(
            urdf_path,
            package_dirs,
            init_q,
            orientation_cost,
            damp,
            use_in_lp_filter,
            in_lp_alpha,
            joint_names_to_lock=[
                "gripper_joint",
                "ur_robotiq_85_left_inner_knuckle_joint",
                "ur_robotiq_85_right_inner_knuckle_joint",
                "ur_robotiq_85_right_knuckle_joint",
                "ur_robotiq_85_left_finger_tip_joint",
                "ur_robotiq_85_right_finger_tip_joint",
            ],
            ee_link_name="wrist_3_link",
        )

        self.ik_policies = [
            UR5PinkIK_Pink(
                urdf_path,
                package_dirs,
                init_q,
                orientation_cost,
                damp,
                use_in_lp_filter,
                in_lp_alpha,
                joint_names_to_lock=[
                    # "gripper_joint", "ur_robotiq_85_left_inner_knuckle_joint",
                    # "ur_robotiq_85_left_inner_finger_joint", "ur_robotiq_85_right_inner_knuckle_joint",
                    # "ur_robotiq_85_right_inner_finger_joint", "ur_robotiq_85_right_outer_knuckle_joint"
                    "gripper_joint",
                    "ur_robotiq_85_left_inner_knuckle_joint",
                    "ur_robotiq_85_right_inner_knuckle_joint",
                    "ur_robotiq_85_right_knuckle_joint",
                    "ur_robotiq_85_left_finger_tip_joint",
                    "ur_robotiq_85_right_finger_tip_joint",
                ],
                ee_link_name="wrist_3_link",
            )
            for _ in range(env_num)
        ]

    def joint_actions_to_cartesian_actions(self, joint_actions, env_id):
        """
        Input:
        joint_actions: (7,)
        env_id: int
        Output:
        cartesian_actions: (8,)
        """
        robot: pin.RobotWrapper = self.ik_policy_dummy.robot
        q0 = np.zeros_like(robot.q0)
        q0[:] = joint_actions[:6]
        robot.framesForwardKinematics(q0)
        ee_pose = robot.data.oMf[
            robot.model.getFrameId(self.ik_policy_dummy.ee_link_name)
        ]
        cartesian_actions = torch.zeros(8, device=joint_actions.device)
        cartesian_actions[-1] = joint_actions[-1]
        cartesian_actions[0:3] = ee_pose.translation
        cartesian_actions[3:7] = ee_pose.rotation
        return cartesian_actions

    def joint_actions_to_cartesian_actions_batch(self, joint_actions, env_ids):
        """
        Input:
        joint_actions: (21, env_num)
        env_ids: (env_num,)
        """
        return torch.stack(
            [
                self.joint_actions_to_cartesian_actions(joint_actions[i], env_ids[i])
                for i in env_ids
            ]
        )

    def cartesian_actions_to_joint_actions(
        self, cartesian_actions, current_joints, env_id
    ):
        """
        Input:
        cartesian_actions: (8,)
        current_joints: (6,)
        env_id: int
        Output:
        joint_actions: (7,)
        """
        device = cartesian_actions.device
        joints_actions = torch.zeros(7, device=device)
        joints_actions[-1] = cartesian_actions[-1]

        target_transform = math_utils.make_pose(
            cartesian_actions[0:3], math_utils.matrix_from_quat(cartesian_actions[3:7])
        )

        lift = cartesian_actions[0]

        joints = self.ik_policies[env_id].solve(target_transform, current_joints)
        joints_actions[:6] = joints
        return joints_actions

    def cartesian_actions_to_joint_actions_batch(
        self, cartesian_actions, current_joints, env_ids
    ):
        """
        Input:
        cartesian_actions: (19, env_num)
        current_left_joints: (7, env_num)
        current_right_joints: (7, env_num)
        env_ids: (env_num,)
        """
        return torch.stack(
            [
                self.cartesian_actions_to_joint_actions(
                    cartesian_actions[i],
                    current_joints[i],
                    env_ids[i],
                )
                for i in env_ids
            ]
        )


class CartesianHelper_Curobo:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: list,
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 0.5,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        env_num: int = 1,
    ):
        self.ik_policy = UR5PinkIK_Curobo(
            urdf_path,
            package_dirs,
            init_q,
            orientation_cost,
            damp,
            use_in_lp_filter,
            in_lp_alpha,
            joint_names_to_lock=[
                "gripper_joint",
                "ur_robotiq_85_left_inner_knuckle_joint",
                "ur_robotiq_85_right_inner_knuckle_joint",
                "ur_robotiq_85_right_knuckle_joint",
                "ur_robotiq_85_left_finger_tip_joint",
                "ur_robotiq_85_right_finger_tip_joint",
            ],
            ee_link_name="wrist_3_link",
        )

    def joint_actions_to_cartesian_actions(self, joint_actions, env_id):
        self.joint_actions_to_cartesian_actions_batch(joint_actions.unsqueeze(0), [0])

    def cartesian_actions_to_joint_actions(
        self, cartesian_actions, current_left_joints, env_id
    ):
        self.cartesian_actions_to_joint_actions_batch(
            cartesian_actions.unsqueeze(0),
            current_left_joints.unsqueeze(0),
            env_ids=[0],
        )

    def joint_actions_to_cartesian_actions_batch(self, joint_actions, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(joint_actions.shape[0])
        assert len(env_ids) == len(joint_actions), "env ids may be wrong"
        joints = joint_actions[:, :6]

        out = self.ik_policy.kin_model.get_state(joints)
        ret_pos = out.ee_position
        ret_quat = out.ee_quaternion
        cartesian_actions = torch.zeros(
            (joint_actions.shape[0], 8), device=joint_actions.device
        )
        cartesian_actions[:, :3] = ret_pos
        cartesian_actions[:, 3:7] = ret_quat
        cartesian_actions[:, 7] = joint_actions[:, 6]
        return cartesian_actions

    def cartesian_actions_to_joint_actions_batch(
        self, cartesian_actions, current_joints, env_ids
    ):
        if env_ids is None:
            env_ids = torch.arange(cartesian_actions.shape[0])

        device = cartesian_actions.device
        if cartesian_actions.device.type == "cpu":
            device = "cuda"
            cartesian_actions = cartesian_actions.to(device)
            current_joints = current_joints.to(device)

        joints_actions = torch.zeros((len(env_ids), 7), device=device)
        q_solution = self.ik_policy.solve_batch(
            cartesian_actions[:, :3],
            cartesian_actions[:, 3:7],
            current_joints[:, :6],
        )
        joints_actions[:, :6] = q_solution
        joints_actions[:, 6] = cartesian_actions[:, 7]
        return joints_actions


class CartesianHelper:
    def __init__(
        self,
        urdf_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf",
        package_dirs: list = [f"{PROJECT_ROOT}/source/arxx7_assets/"],
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 0.5,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        env_num: int = 1,
        use_pink=False,
        device="cuda",
    ):
        self.use_pink = use_pink
        if use_pink:
            self._helper = CartesianHelper_Pink(
                urdf_path,
                package_dirs,
                init_q,
                orientation_cost,
                damp,
                use_in_lp_filter,
                in_lp_alpha,
                env_num,
            )
        else:
            self._helper = CartesianHelper_Curobo(
                urdf_path,
                package_dirs,
                init_q,
                orientation_cost,
                damp,
                use_in_lp_filter,
                in_lp_alpha,
                env_num,
            )

    def joint_actions_to_cartesian_actions(self, joint_actions, env_id):
        return self._helper.joint_actions_to_cartesian_actions(joint_actions, env_id)

    def cartesian_actions_to_joint_actions(
        self, cartesian_actions, current_joints, env_id
    ):
        return self._helper.cartesian_actions_to_joint_actions(
            cartesian_actions, current_joints, env_id
        )

    def joint_actions_to_cartesian_actions_batch(self, joint_actions, env_ids):
        return self._helper.joint_actions_to_cartesian_actions_batch(
            joint_actions, env_ids
        )

    def cartesian_actions_to_joint_actions_batch(
        self, cartesian_actions, current_joints, env_ids
    ):
        return self._helper.cartesian_actions_to_joint_actions_batch(
            cartesian_actions, current_joints, env_ids
        )
