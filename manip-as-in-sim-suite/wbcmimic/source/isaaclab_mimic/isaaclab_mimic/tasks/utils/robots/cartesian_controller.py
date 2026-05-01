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
from isaaclab_mimic.tasks.utils.path import PROJECT_ROOT
from isaaclab_mimic.tasks.utils.robot_ik import X7PinkIKInputLift
from loguru import logger
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask


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
    ):

        self.ik_policies = [
            X7PinkIKInputLift(
                urdf_path,
                package_dirs,
                init_q,
                orientation_cost,
                damp,
                use_in_lp_filter,
                in_lp_alpha,
            )
            for _ in range(env_num)
        ]

    def joint_actions_to_cartesian_actions(self, joint_actions, env_id):
        """
        Input:
        joint_actions: (21,)
        env_id: int
        Output:
        cartesian_actions: (19,)
        """
        full_robot: pin.RobotWrapper = self.ik_policies[env_id].full_robot
        q0 = np.zeros_like(full_robot.q0)
        q0[0] = joint_actions[0]
        q0[2:9] = joint_actions[12:19]
        q0[13:20] = joint_actions[3:10]
        full_robot.framesForwardKinematics(q0)
        ee_pose_l = full_robot.data.oMf[full_robot.model.getFrameId("link10")]
        ee_pose_r = full_robot.data.oMf[full_robot.model.getFrameId("link19")]
        cartesian_actions = torch.zeros(19, device=joint_actions.device)
        cartesian_actions[:3] = joint_actions[:3]
        cartesian_actions[3:6] = ee_pose_l.translation
        cartesian_actions[6:10] = ee_pose_l.rotation
        cartesian_actions[10:13] = ee_pose_r.translation
        cartesian_actions[13:17] = ee_pose_r.rotation
        cartesian_actions[17] = joint_actions[10]
        cartesian_actions[18] = joint_actions[19]
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
        self, cartesian_actions, current_left_joints, current_right_joints, env_id
    ):
        """
        Input:
        cartesian_actions: (19,)
        current_left_joints: (7,)
        current_right_joints: (7,)
        env_id: int
        Output:
        joint_actions: (21,)
        """
        device = cartesian_actions.device
        joints_actions = torch.zeros(21, device=device)
        joints_actions[:3] = cartesian_actions[:3]

        target_left_transform = math_utils.make_pose(
            cartesian_actions[3:6], math_utils.matrix_from_quat(cartesian_actions[6:10])
        )
        target_right_transform = math_utils.make_pose(
            cartesian_actions[10:13],
            math_utils.matrix_from_quat(cartesian_actions[13:17]),
        )

        lift = cartesian_actions[0]

        left_joints, right_joints = self.ik_policies[env_id].solve(
            target_left_transform,
            target_right_transform,
            current_left_joints,
            current_right_joints,
            lift,
        )
        joints_actions[3:10] = left_joints
        joints_actions[10] = cartesian_actions[17]
        joints_actions[11] = cartesian_actions[17]
        joints_actions[12:19] = right_joints
        joints_actions[19] = cartesian_actions[18]
        joints_actions[20] = cartesian_actions[18]
        return joints_actions

    def cartesian_actions_to_joint_actions_batch(
        self, cartesian_actions, current_left_joints, current_right_joints, env_ids
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
                    current_left_joints[i],
                    current_right_joints[i],
                    env_ids[i],
                )
                for i in env_ids
            ]
        )
