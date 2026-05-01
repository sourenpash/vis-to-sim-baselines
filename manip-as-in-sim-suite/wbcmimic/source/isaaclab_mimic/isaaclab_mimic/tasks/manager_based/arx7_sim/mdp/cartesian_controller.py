# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import random
import time
from multiprocessing import Pipe, Process
from pathlib import Path

import isaaclab.utils.math as math_utils
import numpy as np
import pink
import pinocchio as pin
import qpsolvers
import torch
from isaaclab_mimic.utils.path import PROJECT_ROOT
from isaaclab_mimic.utils.robot_ik import X7IKCurobo, X7PinkIKInputLift, X7IKCuroboInputLift
from loguru import logger
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask

# multiprocessing.set_start_method("spawn", force=True)

# def run_ik_in_process(
#     urdf_path,
#     package_dirs,
#     init_q,
#     orientation_cost,
#     damp,
#     use_in_lp_filter,
#     in_lp_alpha,
#     connection,
# ):
#     ik_policy = X7PinkIKInputLift(
#         urdf_path,
#         package_dirs,
#         init_q,
#         orientation_cost,
#         damp,
#         use_in_lp_filter,
#         in_lp_alpha,
#     )
#     full_robot: pin.RobotWrapper = ik_policy.full_robot
#     while True:
#         data = connection.recv()
#         if data.get("fk", None) is not None:
#             full_robot.framesForwardKinematics(data["fk"]["q0"])
#             ee_pose_l = full_robot.data.oMf[full_robot.model.getFrameId("link10")]
#             ee_pose_r = full_robot.data.oMf[full_robot.model.getFrameId("link19")]
#             cartesian_actions = torch.zeros(19, device=data["fk"]["device"])
#             cartesian_actions[3:6] = ee_pose_l.translation
#             cartesian_actions[6:10] = ee_pose_l.rotation
#             cartesian_actions[10:13] = ee_pose_r.translation
#             cartesian_actions[13:17] = ee_pose_r.rotation
#             connection.send({"fk": cartesian_actions})
#         elif data.get("ik", None) is not None:
#             cartesian_actions = data["ik"]["cartesian_actions"]
#             current_left_joints = data["ik"]["current_left_joints"]
#             current_right_joints = data["ik"]["current_right_joints"]
#             device = cartesian_actions.device
#             joints_actions = torch.zeros(21, device=device)
#             joints_actions[:3] = cartesian_actions[:3]

#             target_left_transform = math_utils.make_pose(
#                 cartesian_actions[3:6], math_utils.matrix_from_quat(cartesian_actions[6:10])
#             )
#             target_right_transform = math_utils.make_pose(
#                 cartesian_actions[10:13],
#                 math_utils.matrix_from_quat(cartesian_actions[13:17]),
#             )
#             lift = cartesian_actions[0]
#             left_joints, right_joints = ik_policy.solve(
#                 target_left_transform,
#                 target_right_transform,
#                 current_left_joints,
#                 current_right_joints,
#                 lift,
#             )
#             joints_actions[3:10] = left_joints
#             joints_actions[10] = cartesian_actions[17]
#             joints_actions[11] = cartesian_actions[17]
#             joints_actions[12:19] = right_joints
#             joints_actions[19] = cartesian_actions[18]
#             joints_actions[20] = cartesian_actions[18]
#             connection.send({"ik": joints_actions})

# class CartesianHelper_Pink:
#     def __init__(
#         self,
#         urdf_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf",
#         package_dirs: list = [f"{PROJECT_ROOT}/source/arxx7_assets/"],
#         init_q: np.ndarray = None,  # 允许传入初始关节位置
#         orientation_cost: float = 0.5,
#         damp: float = 1e-6,
#         use_in_lp_filter: bool = True,
#         in_lp_alpha: float = 0.9,
#         env_num: int = 1,
#     ):
#         self.ik_policy_dummy = X7PinkIKInputLift(
#             urdf_path,
#             package_dirs,
#             init_q,
#             orientation_cost,
#             damp,
#             use_in_lp_filter,
#             in_lp_alpha,
#         )

#         self.ik_policy_pipes = []
#         self.ik_policy_processes = []
#         for i in range(env_num):
#             parent_conn, child_conn = Pipe()
#             process = Process(
#                 target=run_ik_in_process,
#                 args=(
#                     urdf_path,
#                     package_dirs,
#                     init_q,
#                     orientation_cost,
#                     damp,
#                     use_in_lp_filter,
#                     in_lp_alpha,
#                     child_conn,
#                 ),
#             )
#             process.start()
#             self.ik_policy_pipes.append(parent_conn)
#             self.ik_policy_processes.append(process)

#     def joint_actions_to_cartesian_actions(self, joint_actions, env_id):
#         """
#         Input:
#         joint_actions: (21,)
#         env_id: int
#         Output:
#         cartesian_actions: (19,)
#         """
#         full_robot: pin.RobotWrapper = self.ik_policy_dummy.full_robot
#         q0 = np.zeros_like(full_robot.q0)
#         q0[0] = joint_actions[0]
#         q0[2:9] = joint_actions[12:19]
#         q0[13:20] = joint_actions[3:10]
#         data = {
#             "fk": {
#                 "q0": q0,
#                 "device": joint_actions.device,
#             }
#         }
#         self.ik_policy_pipes[env_id].send(data)
#         cartesian_actions = self.ik_policy_pipes[env_id].recv()["fk"]
#         cartesian_actions[:3] = joint_actions[:3]
#         cartesian_actions[17] = joint_actions[10]
#         cartesian_actions[18] = joint_actions[19]
#         return cartesian_actions

#     def joint_actions_to_cartesian_actions_batch(self, joint_actions, env_ids):
#         """
#         Input:
#         joint_actions: (21, env_num)
#         env_ids: (env_num,)
#         """
#         cartesian_actions_lst = []
#         full_robot: pin.RobotWrapper = self.ik_policy_dummy.full_robot
#         for i in env_ids:
#             q0 = np.zeros_like(full_robot.q0)
#             q0[0] = joint_actions[i, 0]
#             q0[2:9] = joint_actions[i, 12:19]
#             q0[13:20] = joint_actions[i, 3:10]
#             data = {
#                 "fk": {
#                     "q0": q0,
#                     "device": joint_actions.device,
#                 }
#             }
#             self.ik_policy_pipes[i].send(data)

#         for i in env_ids:
#             cartesian_actions = self.ik_policy_pipes[i].recv()["fk"]
#             cartesian_actions[:3] = joint_actions[i, :3]
#             cartesian_actions[17] = joint_actions[i, 10]
#             cartesian_actions[18] = joint_actions[i, 19]
#             cartesian_actions_lst.append(cartesian_actions)
#         return torch.stack(cartesian_actions_lst)

#     def cartesian_actions_to_joint_actions(
#         self, cartesian_actions, current_left_joints, current_right_joints, env_id
#     ):
#         """
#         Input:
#         cartesian_actions: (19,)
#         current_left_joints: (7,)
#         current_right_joints: (7,)
#         env_id: int
#         Output:
#         joint_actions: (21,)
#         """
#         data = {
#             "ik": {
#                 "cartesian_actions": cartesian_actions,
#                 "current_left_joints": current_left_joints,
#                 "current_right_joints": current_right_joints,
#             }
#         }
#         self.ik_policy_pipes[env_id].send(data)
#         joints_actions = self.ik_policy_pipes[env_id].recv()["ik"]
#         return joints_actions

#     def cartesian_actions_to_joint_actions_batch(
#         self, cartesian_actions, current_left_joints, current_right_joints, env_ids
#     ):
#         """
#         Input:
#         cartesian_actions: (19, env_num)
#         current_left_joints: (7, env_num)
#         current_right_joints: (7, env_num)
#         env_ids: (env_num,)
#         """
#         joint_actions_lst = []
#         for i in env_ids:
#             data = {
#                 "ik": {
#                     "cartesian_actions": cartesian_actions[i],
#                     "current_left_joints": current_left_joints[i],
#                     "current_right_joints": current_right_joints[i],
#                 }
#             }
#             self.ik_policy_pipes[i].send(data)
#         for i in env_ids:
#             joints_actions = self.ik_policy_pipes[i].recv()["ik"]
#             joint_actions_lst.append(joints_actions)
#         return torch.stack(joint_actions_lst)


class CartesianHelper_Pink:
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
        self.ik_policy_dummy = X7PinkIKInputLift(
            urdf_path,
            package_dirs,
            init_q,
            orientation_cost,
            damp,
            use_in_lp_filter,
            in_lp_alpha,
        )

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


class CartesianHelper_Curobo:
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
        device="cuda",
        use_ik_solve_lift: bool = True,
    ):
        self.use_ik_solve_lift = use_ik_solve_lift
        if use_ik_solve_lift:
            self.ik = X7IKCurobo(
                urdf_path,
                package_dirs,
                init_q,
                orientation_cost,
                damp,
                use_in_lp_filter,
                in_lp_alpha,
                device=device,
            )
            self.left_joint_idxes = [3, 4, 5, 6, 7, 8, 9]
            self.right_joint_idxes = [12, 13, 14, 15, 16, 17, 18]
        else:
            urdf_path: Path = Path(urdf_path)
            urdf_path_fix_lift = urdf_path.parent / f"{urdf_path.stem}_fix_lift.urdf"
            self.ik = X7IKCuroboInputLift(
                urdf_path_fix_lift,
                package_dirs,
                init_q,
                orientation_cost,
                damp,
                use_in_lp_filter,
                in_lp_alpha,
                device=device,
            )

    def joint_actions_to_cartesian_actions(self, joint_actions, env_id):
        self.joint_actions_to_cartesian_actions_batch(joint_actions.unsqueeze(0), [0])

    def cartesian_actions_to_joint_actions(
        self, cartesian_actions, current_left_joints, current_right_joints, env_id
    ):
        """Convert cartesian space actions to joint space actions for a single environment.

        Args:
            cartesian_actions: (19,) tensor containing cartesian space actions
            current_left_joints: (7,) tensor of current left arm joint positions
            current_right_joints: (7,) tensor of current right arm joint positions
            env_id: Environment ID to specify which robot to control

        Returns:
            Joint space actions (21,) tensor by calling the batch version of this method
        """
        self.cartesian_actions_to_joint_actions_batch(
            cartesian_actions.unsqueeze(0),
            current_left_joints.unsqueeze(0),
            current_left_joints.unsqueeze(0),
            env_ids=[0],
        )

    def joint_actions_to_cartesian_actions_batch(self, joint_actions, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(joint_actions.shape[0])
        assert len(env_ids) == len(joint_actions), "env ids may be wrong"
        left_joint = joint_actions[:, self.left_joint_idxes]
        right_joint = joint_actions[:, self.right_joint_idxes]
        left_ret = self.ik.left_kin_model.get_state(left_joint)
        right_ret = self.ik.left_kin_model.get_state(right_joint)
        left_ret_pos = left_ret.ee_position
        left_ret_quat = left_ret.ee_quaternion
        right_ret_pos = right_ret.ee_position
        right_ret_quat = right_ret.ee_quaternion
        cartesian_actions = torch.zeros(
            (joint_actions.shape[0], 19), device=joint_actions.device
        )
        cartesian_actions[:, :3] = joint_actions[:, :3]
        cartesian_actions[:, 3:6] = left_ret_pos
        cartesian_actions[6:10] = left_ret_quat
        cartesian_actions[10:13] = right_ret_pos
        cartesian_actions[13:17] = right_ret_quat
        cartesian_actions[:, 17] = joint_actions[:, 10]
        cartesian_actions[:, 18] = joint_actions[:, 19]
        return cartesian_actions

    def cartesian_actions_to_joint_actions_batch(
        self, cartesian_actions, current_left_joints, current_right_joints, env_ids=None
    ):
        if env_ids is None:
            env_ids = torch.arange(cartesian_actions.shape[0])

        device = "cuda"  # temp for now
        joints_actions = torch.zeros((len(env_ids), 21), device=device)
        joints_actions[:, :3] = cartesian_actions[:, :3]

        target_left_transform = math_utils.make_pose(
            cartesian_actions[:, 3:6],
            math_utils.matrix_from_quat(cartesian_actions[:, 6:10]),
        )
        target_right_transform = math_utils.make_pose(
            cartesian_actions[:, 10:13],
            math_utils.matrix_from_quat(cartesian_actions[:, 13:17]),
        )

        lift = cartesian_actions[:, 0:1]
        origin_device = target_left_transform.device
        left_joints, right_joints = self.ik.solve(
            target_left_transform.to(device),
            target_right_transform.to(device),
            current_left_joints.to(device),
            current_right_joints.to(device),
            lift.to(device),
        )
        
        if self.use_ik_solve_lift:
            joints_actions[:, 3:10] = left_joints[:, 1:]
            joints_actions[:, 12:19] = right_joints[:, 1:]
        else:
            joints_actions[:, 3:10] = left_joints[:, :]
            joints_actions[:, 12:19] = right_joints[:, :]
        joints_actions[:, 10] = cartesian_actions[:, 17]
        joints_actions[:, 11] = cartesian_actions[:, 17]
        joints_actions[:, 19] = cartesian_actions[:, 18]
        joints_actions[:, 20] = cartesian_actions[:, 18]
        return joints_actions.to(origin_device)


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
        use_ik_solve_lift: bool = True,
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
                device,
                use_ik_solve_lift=use_ik_solve_lift,
            )

    def joint_actions_to_cartesian_actions(self, joint_actions, env_id):
        return self._helper.joint_actions_to_cartesian_actions(joint_actions, env_id)

    def cartesian_actions_to_joint_actions(
        self, cartesian_actions, current_left_joints, current_right_joints, env_id
    ):
        return self._helper.cartesian_actions_to_joint_actions(
            cartesian_actions, current_left_joints, current_right_joints, env_id
        )

    def joint_actions_to_cartesian_actions_batch(self, joint_actions, env_ids):
        return self._helper.joint_actions_to_cartesian_actions_batch(
            joint_actions, env_ids
        )

    def cartesian_actions_to_joint_actions_batch(
        self, cartesian_actions, current_left_joints, current_right_joints, env_ids
    ):
        return self._helper.cartesian_actions_to_joint_actions_batch(
            cartesian_actions, current_left_joints, current_right_joints, env_ids
        )
