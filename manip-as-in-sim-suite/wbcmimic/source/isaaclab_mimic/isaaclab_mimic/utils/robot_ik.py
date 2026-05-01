# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from typing import List

import numpy as np
import pink
import pinocchio as pin
import qpsolvers
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_logger
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from loguru import logger
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask

from .path import PROJECT_ROOT

setup_logger(level="error")


class LPConstrainedSE3Filter:
    def __init__(self, alpha, dt=1 / 60):
        self.alpha = alpha
        self.dt = dt
        self.is_init = False

        self.prev_pos = None
        self.prev_vel = None

        self.max_vel = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
        self.max_acc = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3])

    def next(self, target):
        if not self.is_init:
            self.prev_pos = pin.SE3(target)
            self.prev_vel = np.zeros(6)
            self.prev_acc = np.zeros(6)
            self.is_init = True
            return np.array(self.prev_pos)

        ip_pos = pin.SE3.Interpolate(self.prev_pos, pin.SE3(target), self.alpha)
        ip_vel = pin.log(ip_pos.actInv(self.prev_pos)).vector / self.dt
        ip_acc = (ip_vel - self.prev_vel) / self.dt

        acc = np.clip(ip_acc, -self.max_acc, self.max_acc)
        vel = np.clip(self.prev_vel + acc * self.dt, -self.max_vel, self.max_vel)
        pos = self.prev_pos * (
            ~pin.exp(vel * self.dt)
        )  # Caution! * means matrix multiplication in pinocchio

        self.prev_pos = pos
        self.prev_vel = vel

        return np.array(self.prev_pos)

    def reset(self):
        self.y = None
        self.is_init = False


class X7PinkIK:
    def __init__(
        self,
        urdf_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf",
        package_dirs: list = [f"{PROJECT_ROOT}/source/arxx7_assets/"],
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 1.0,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
    ):
        # 加载完整机器人模型
        self.full_robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=package_dirs,
            root_joint=None,
        )

        # 保存关节信息
        self.joint_names = [name for name in self.full_robot.model.names]
        self.active_joint_ids = []
        self.locked_joint_ids = []

        # 确定活动关节和锁定关节
        for joint_id in range(1, self.full_robot.model.njoints):  # 跳过universe关节
            joint_name = self.full_robot.model.names[joint_id]
            if joint_name.startswith("joint"):
                joint_num = int(joint_name[5:])
                if 4 <= joint_num <= 10 or 13 <= joint_num <= 19:
                    self.active_joint_ids.append(joint_id)
                else:
                    self.locked_joint_ids.append(joint_id)
            else:
                self.locked_joint_ids.append(joint_id)

        # 保存完整机器人的默认配置
        self.full_q0 = np.copy(self.full_robot.q0)

        # 如果提供了初始姿态，使用它来更新默认配置
        if init_q is not None:
            assert len(init_q) == len(
                self.full_q0
            ), f"初始位置长度不匹配: {len(init_q)} vs {len(self.full_q0)}"
            self.full_q0 = np.copy(init_q)

        # 构建精简模型
        self.robot = self.full_robot.buildReducedRobot(self.locked_joint_ids)

        # 映射从完整机器人到精简机器人的关节位置
        self.reduced_indices = []
        reduced_joint_names = [
            self.robot.model.names[i] for i in range(self.robot.model.njoints)
        ]
        for joint_id in self.active_joint_ids:
            joint_name = self.full_robot.model.names[joint_id]
            if joint_name in reduced_joint_names:
                self.reduced_indices.append(reduced_joint_names.index(joint_name))

        # 创建适用于精简模型的初始配置
        self.reduced_q0 = np.copy(self.robot.q0)

        # 从完整配置更新精简配置中的活动关节
        self._update_reduced_q_from_full(self.full_q0)

        # 初始化配置
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

        # 初始化任务
        self.right_hand_task = FrameTask(
            "link19",  # 需根据实际URDF调整
            position_cost=1.0,
            orientation_cost=orientation_cost,
        )
        self.left_hand_task = FrameTask(
            "link10",  # 需根据实际URDF调整
            position_cost=1.0,
            orientation_cost=orientation_cost,
        )
        self.posture_task = PostureTask(cost=1e-2)

        # 从当前配置设置姿态任务目标
        self.posture_task.set_target_from_configuration(self.configuration)

        self.tasks = [
            self.right_hand_task,
            self.left_hand_task,
            self.posture_task,
        ]

        # QP求解器配置
        self.solver = qpsolvers.available_solvers[0]
        self.dt = 0.01
        self.damp = damp

        # 初始化滤波器
        if use_in_lp_filter:
            self.right_filter = LPConstrainedSE3Filter(in_lp_alpha)
            self.left_filter = LPConstrainedSE3Filter(in_lp_alpha)
        else:
            self.right_filter = None
            self.left_filter = None

        # 输出关节信息
        print(f"活动关节数量: {len(self.active_joint_ids)}")
        print(
            f"活动关节: {[self.full_robot.model.names[i] for i in self.active_joint_ids]}"
        )

    def _update_reduced_q_from_full(self, full_q):
        """从完整配置更新精简模型配置"""
        # 遍历活动关节，将值从完整配置复制到精简配置
        for i, joint_id in enumerate(self.active_joint_ids):
            joint_name = self.full_robot.model.names[joint_id]
            joint_idx = self.full_robot.model.getJointId(joint_name)

            # 获取关节在完整模型配置中的索引范围
            joint = self.full_robot.model.joints[joint_idx]
            q_idx = self.full_robot.model.idx_qs[joint_idx]
            nq = joint.nq

            # 获取关节在精简模型中的索引
            reduced_idx = self.robot.model.getJointId(joint_name)
            reduced_q_idx = self.robot.model.idx_qs[reduced_idx]

            # 复制配置
            self.reduced_q0[reduced_q_idx : reduced_q_idx + nq] = full_q[
                q_idx : q_idx + nq
            ]

    def set_init_position(self, init_q):
        """设置新的初始位置"""
        assert len(init_q) == len(
            self.full_q0
        ), f"初始位置长度不匹配: {len(init_q)} vs {len(self.full_q0)}"
        self.full_q0 = np.copy(init_q)

        # 更新精简模型的初始配置
        self._update_reduced_q_from_full(self.full_q0)

        # 更新当前配置
        self.configuration.q = np.copy(self.reduced_q0)

        # 重置任务目标
        self.posture_task.set_target_from_configuration(self.configuration)

        # 重置滤波器
        if self.right_filter is not None:
            self.right_filter.reset()
        if self.left_filter is not None:
            self.left_filter.reset()

        return self.configuration.q

    def solve(
        self,
        left_wrist_transform,
        right_wrist_transform,
        joint_4_to_10,
        joint_13_to_19,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """求解双臂逆运动学

        Args:
            left_wrist_transform: 4x4 齐次变换矩阵，表示左手腕目标位姿
            right_wrist_transform: 4x4 齐次变换矩阵，表示右手腕目标位姿
            orientation_cost: 可选方向权重覆盖值

        Returns:
            (左臂关节位置, 右臂关节位置)
        """
        # 更新方向权重
        device = "cuda"
        if isinstance(joint_13_to_19, torch.Tensor):
            device = joint_13_to_19.device
            joint_13_to_19 = joint_13_to_19.cpu().numpy()
        if isinstance(joint_4_to_10, torch.Tensor):
            device = joint_4_to_10.device
            joint_4_to_10 = joint_4_to_10.cpu().numpy()
        if isinstance(right_wrist_transform, torch.Tensor):
            device = right_wrist_transform.device
            right_wrist_transform = right_wrist_transform.cpu().numpy()
        if isinstance(left_wrist_transform, torch.Tensor):
            device = left_wrist_transform.device
            left_wrist_transform = left_wrist_transform.cpu().numpy()
        if orientation_cost is not None:
            self.right_hand_task.orientation_cost = orientation_cost
            self.left_hand_task.orientation_cost = orientation_cost

        # 应用平滑滤波
        if self.right_filter is not None:
            right_transform = self.right_filter.next(right_wrist_transform)
        else:
            right_transform = right_wrist_transform
        if self.left_filter is not None:
            left_transform = self.left_filter.next(left_wrist_transform)
        else:
            left_transform = left_wrist_transform

        self.reduced_q0 = np.concatenate([joint_13_to_19, joint_4_to_10])
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

        # 设置任务目标
        self.right_hand_task.set_target(pin.SE3(right_transform))
        self.left_hand_task.set_target(pin.SE3(left_transform))

        # velocity = solve_ik(self.configuration, self.tasks, self.dt, solver="quadprog", damping=self.damp, safety_break=False)
        # self.configuration.integrate_inplace(velocity, self.dt)

        initial_error = np.linalg.norm(
            self.left_hand_task.compute_error(self.configuration)
        ) + np.linalg.norm(self.right_hand_task.compute_error(self.configuration))
        error_norm = initial_error

        logger.trace(f"Initial error norm: {error_norm}")

        # # 求解IK
        # velocity = solve_ik(
        #     self.configuration,
        #     self.tasks,
        #     self.dt,
        #     solver=self.solver,
        #     damping=self.damp,
        # )
        # self.configuration.integrate_inplace(velocity, self.dt)

        # # 假设右臂关节是前7个，左臂关节是后7个
        # # 注意: 根据实际机器人配置可能需要调整
        # qpos = self.configuration.q
        # right_arm_indices = list(range(0, 7))
        # left_arm_indices = list(range(7, 14))

        # return qpos[right_arm_indices], qpos[left_arm_indices], 0.0

        nb_steps = 0
        damping_factor = 1e-8  #  1e-3
        prev_error = float("inf")
        current_dt = dt
        try:
            while error_norm > stop_thres and nb_steps < max_try_times:
                dv = solve_ik(
                    self.configuration,
                    tasks=self.tasks,
                    dt=current_dt,
                    damping=damping_factor,
                    safety_break=False,
                    solver="quadprog",
                )
                if nb_steps % 5 == 0:
                    damping_factor *= 1.5
                # 如果误差减小不明显，减小步长
                if prev_error - error_norm < error_norm * 0.01:
                    current_dt *= 0.8
                prev_error = error_norm
                q_out = pin.integrate(
                    self.robot.model, self.configuration.q, dv * current_dt
                )
                self.configuration = pink.Configuration(
                    self.robot.model, self.robot.data, q_out
                )
                pin.updateFramePlacements(self.robot.model, self.robot.data)
                error_norm = np.linalg.norm(
                    self.left_hand_task.compute_error(self.configuration)
                ) + np.linalg.norm(
                    self.right_hand_task.compute_error(self.configuration)
                )
                nb_steps += 1
        except Exception as e:
            logger.error(f"IK failed: {e}")
            return torch.from_numpy(self.reduced_q0[7:14]).to(device), torch.from_numpy(
                self.reduced_q0[:7]
            ).to(device)

        # print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
        return torch.from_numpy(self.configuration.q[7:14]).to(
            device
        ), torch.from_numpy(self.configuration.q[:7]).to(device)

    def solve_single_arm(
        self,
        wrist_transform,
        joints,
        left=True,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """求解单臂逆运动学

        Args:
            wrist_transform: 4x4 齐次变换矩阵，表示手腕目标位姿
            joints: 关节索引列表
            left: 是否求解左手
            orientation_cost: 可选方向权重覆盖值

        Returns:
            (右臂关节位置, 左臂关节位置, 误差)
        """
        # 更新方向权重
        device = "cuda"
        if isinstance(joints, torch.Tensor):
            device = joints.device
            joints = joints.cpu().numpy()
        if isinstance(wrist_transform, torch.Tensor):
            device = wrist_transform.device
            wrist_transform = wrist_transform.cpu().numpy()
        if orientation_cost is not None:
            self.right_hand_task.orientation_cost = orientation_cost
            self.left_hand_task.orientation_cost = orientation_cost

        # 应用平滑滤波
        if self.right_filter and not left:
            wrist_transform = self.right_filter.next(wrist_transform)
        elif self.left_filter and left:
            wrist_transform = self.left_filter.next(wrist_transform)
        else:
            wrist_transform = wrist_transform

        if left:
            joint_4_to_10 = joints
            joint_13_to_19 = self.reduced_q0[:7]
        else:
            joint_13_to_19 = joints
            joint_4_to_10 = self.reduced_q0[7:]

        self.reduced_q0 = np.concatenate([joint_13_to_19, joint_4_to_10])
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

        # 设置任务目标
        self.left_hand_task.set_target(pin.SE3(wrist_transform))
        self.right_hand_task.set_target(pin.SE3(wrist_transform))

        if left:
            error_norm = np.linalg.norm(
                self.left_hand_task.compute_error(self.configuration)
            )
        else:
            error_norm = np.linalg.norm(
                self.right_hand_task.compute_error(self.configuration)
            )

        logger.trace(f"Initial error norm: {error_norm}")

        # # 求解IK
        # velocity = solve_ik(
        #     self.configuration,
        #     self.tasks,
        #     self.dt,
        #     solver=self.solver,
        #     damping=self.damp,
        # )
        # self.configuration.integrate_inplace(velocity, self.dt)

        # # 假设右臂关节是前7个，左臂关节是后7个
        # # 注意: 根据实际机器人配置可能需要调整
        # qpos = self.configuration.q
        # right_arm_indices = list(range(0, 7))
        # left_arm_indices = list(range(7, 14))

        # return qpos[right_arm_indices], qpos[left_arm_indices], 0.0

        nb_steps = 0

        while error_norm > stop_thres and nb_steps < max_try_times:
            dv = solve_ik(
                self.configuration,
                tasks=self.tasks,
                dt=dt,
                damping=1e-8,
                solver="quadprog",
                safety_break=False,
            )
            q_out = pin.integrate(self.robot.model, self.configuration.q, dv * dt)
            self.configuration = pink.Configuration(
                self.robot.model, self.robot.data, q_out
            )
            pin.updateFramePlacements(self.robot.model, self.robot.data)
            if left:
                error_norm = np.linalg.norm(
                    self.left_hand_task.compute_error(self.configuration)
                )
            else:
                error_norm = np.linalg.norm(
                    self.right_hand_task.compute_error(self.configuration)
                )
                nb_steps += 1

        # print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
        if left:
            return torch.from_numpy(self.configuration.q[7:14]).to(device)
        else:
            return torch.from_numpy(self.configuration.q[:7]).to(device)


class X7PinkIKInputLift(X7PinkIK):

    def update_reduced_robot(self, full_q0):
        self.full_q0 = np.copy(full_q0)
        self.robot = self.full_robot.buildReducedRobot(
            self.locked_joint_ids, self.full_q0
        )
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

    def solve(
        self,
        left_wrist_transform,
        right_wrist_transform,
        joint_4_to_10,
        joint_13_to_19,
        lift,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        if lift != self.full_q0[0]:
            self.full_q0[0] = lift
            self.update_reduced_robot(self.full_q0)
        res = super().solve(
            left_wrist_transform,
            right_wrist_transform,
            joint_4_to_10,
            joint_13_to_19,
            orientation_cost,
            stop_thres,
            dt,
            max_try_times,
        )
        return res
        # left_q = self.solve_single_arm(left_wrist_transform, joint_4_to_10, lift, left=True, orientation_cost=orientation_cost, stop_thres=stop_thres, dt=dt, max_try_times=max_try_times)
        # right_q = self.solve_single_arm(right_wrist_transform, joint_13_to_19, lift, left=False, orientation_cost=orientation_cost, stop_thres=stop_thres, dt=dt, max_try_times=max_try_times)
        # return left_q, right_q

    def solve_single_arm(
        self,
        wrist_transform,
        joints,
        lift,
        left=True,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        if lift != self.full_q0[0]:
            self.full_q0[0] = lift
            self.update_reduced_robot(self.full_q0)
        return super().solve_single_arm(
            wrist_transform,
            joints,
            left,
            orientation_cost,
            stop_thres,
            dt,
            max_try_times,
        )


class X7IKCurobo:
    def __init__(
        self,
        urdf_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf",
        package_dirs: list = [f"{PROJECT_ROOT}/source/arxx7_assets/"],
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 1.0,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        yml_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/curobo/X7.yml",
        device="cuda",
    ):
        self.lr_to_link_name = {"l": "link10", "r": "link19"}
        self.lr_to_index = {
            "l": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device),
            "r": torch.tensor([0, 8, 9, 10, 11, 12, 13, 14], device=device),
        }
        self.tensor_args = TensorDeviceType(device=torch.device(device))
        self.left_ee_link_name = "link10"
        self.right_ee_link_name = "link19"
        self.base_link_name = "base_link"
        self.orientation_cost = orientation_cost
        self.use_in_lp_filter = use_in_lp_filter
        self.in_lp_alpha = in_lp_alpha
        self.filter_state = None

        self.left_arm_robot_cfg = RobotConfig.from_basic(
            urdf_path, self.base_link_name, self.left_ee_link_name, self.tensor_args
        )
        self.right_arm_robot_cfg = RobotConfig.from_basic(
            urdf_path, self.base_link_name, self.right_ee_link_name, self.tensor_args
        )
        self.robot_cfg = RobotConfig.from_dict(load_yaml(yml_path))
        self.left_kin_model = CudaRobotModel(self.left_arm_robot_cfg.kinematics)
        self.right_kin_model = CudaRobotModel(self.right_arm_robot_cfg.kinematics)
        self.robot_kin_model = CudaRobotModel(self.robot_cfg.kinematics)
        self.left_dof = self.left_kin_model.get_dof()
        self.right_dof = self.right_kin_model.get_dof()

        # assert for arx x7
        assert self.left_dof == 8
        assert self.right_dof == 8

        self.left_ik_config = IKSolverConfig.load_from_robot_config(
            self.left_arm_robot_cfg,
            None,
            position_threshold=0.005,
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            high_precision=False,
            use_cuda_graph=True,
            grad_iters=None,
            regularization=False,
        )
        self.right_ik_config = IKSolverConfig.load_from_robot_config(
            self.right_arm_robot_cfg,
            None,
            position_threshold=0.005,
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            high_precision=False,
            use_cuda_graph=True,
            grad_iters=None,
            regularization=False,
        )
        self.robot_ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            position_threshold=0.005,
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            high_precision=False,
            use_cuda_graph=True,
            grad_iters=None,
            regularization=False,
        )
        self.left_ik_solver = IKSolver(self.left_ik_config)
        self.right_ik_solver = IKSolver(self.right_ik_config)
        self.robot_ik_solver = IKSolver(self.robot_ik_config)
        self.last_shape = None
        self.last_dtype = None

    def solve(
        self,
        left_wrist_transform,
        right_wrist_transform,
        joint_4_to_10,
        joint_13_to_19,
        lift,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        if (
            left_wrist_transform.shape != self.last_shape
            or left_wrist_transform.dtype != self.last_dtype
        ):
            self.last_shape = left_wrist_transform.shape
            self.last_dtype = left_wrist_transform.dtype
            infer_mode = False
        else:
            infer_mode = True
        with torch.inference_mode(infer_mode):
            left_pose = Pose(
                position=left_wrist_transform[..., :3, 3],
                rotation=left_wrist_transform[..., :3, :3],
            )
            right_pose = Pose(
                position=right_wrist_transform[..., :3, 3],
                rotation=right_wrist_transform[..., :3, :3],
            )
            joints = torch.cat([lift, joint_4_to_10, joint_13_to_19], dim=-1)
            res = self.robot_ik_solver.solve_batch(
                left_pose,
                retract_config=joints,
                seed_config=joints[None],
                return_seeds=1,
                link_poses={
                    self.lr_to_link_name["l"]: left_pose,
                    self.lr_to_link_name["r"]: right_pose,
                },
            )
            # pay attention to whether res.js_solution_joint_names is right.
        return (
            res.solution[:, 0, self.lr_to_index["l"]],
            res.solution[:, 0, self.lr_to_index["r"]],
        )

    def solve_single_arm(
        self,
        wrist_transform: torch.Tensor,
        joints: torch.Tensor,
        lift: torch.Tensor,
        left=True,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """
        wrist_transform: (num_env x 4 x 4)
        joints: (num_env x 7)
        lift: (num_env x 1)
        """
        if (
            wrist_transform.shape != self.last_shape
            or wrist_transform.dtype != self.last_dtype
        ):
            # this check is now not enough, use after checking it.
            self.last_shape = wrist_transform.shape
            self.last_dtype = wrist_transform.dtype
            infer_mode = False
        else:
            infer_mode = True
        with torch.inference_mode(infer_mode):
            pose = Pose(
                position=wrist_transform[..., :3, 3],
                rotation=wrist_transform[..., :3, :3],
            )
            joints_curobo = torch.cat([lift, joints], dim=-1)
            if left:
                res = self.left_ik_solver.solve_batch(
                    pose,
                    retract_config=joints_curobo,
                    seed_config=joints_curobo[None],
                    return_seeds=1,
                )
            else:
                res = self.right_ik_solver.solve_batch(
                    pose,
                    retract_config=joints_curobo,
                    seed_config=joints_curobo[None],
                    return_seeds=1,
                )
            q_solution: torch.Tensor = res.solution
        return q_solution

class X7IKCuroboInputLift:
    def __init__(
        self,
        urdf_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7_fix_lift.urdf",
        package_dirs: list = [f"{PROJECT_ROOT}/source/arxx7_assets/"],
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 1.0,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        yml_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/curobo/X7.yml",
        device="cuda",
    ):
        self.lr_to_link_name = {"l": "link10", "r": "link19"}
        self.lr_to_index = {
            "l": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device),
            "r": torch.tensor([0, 8, 9, 10, 11, 12, 13, 14], device=device),
        }
        self.tensor_args = TensorDeviceType(device=torch.device(device))
        self.left_ee_link_name = "link10"
        self.right_ee_link_name = "link19"
        self.base_link_name = "base_link"
        self.orientation_cost = orientation_cost
        self.use_in_lp_filter = use_in_lp_filter
        self.in_lp_alpha = in_lp_alpha
        self.filter_state = None

        self.left_arm_robot_cfg = RobotConfig.from_basic(
            urdf_path, self.base_link_name, self.left_ee_link_name, self.tensor_args
        )
        self.right_arm_robot_cfg = RobotConfig.from_basic(
            urdf_path, self.base_link_name, self.right_ee_link_name, self.tensor_args,
        )
        self.left_kin_model = CudaRobotModel(self.left_arm_robot_cfg.kinematics)
        self.right_kin_model = CudaRobotModel(self.right_arm_robot_cfg.kinematics)
        
        self.left_dof = self.left_kin_model.get_dof()
        self.right_dof = self.right_kin_model.get_dof()

        # assert for arx x7
        assert self.left_dof == 7, "lift joint must be fixed so the left arm dof is 7 but get {}".format(self.left_dof)
        assert self.right_dof == 7, "lift joint must be fixed so the right arm dof is 7 but get {}".format(self.right_dof)

        self.left_ik_config = IKSolverConfig.load_from_robot_config(
            self.left_arm_robot_cfg,
            None,
            position_threshold=0.005,
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            high_precision=False,
            use_cuda_graph=True,
            grad_iters=None,
            regularization=False,
        )
        self.right_ik_config = IKSolverConfig.load_from_robot_config(
            self.right_arm_robot_cfg,
            None,
            position_threshold=0.005,
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            high_precision=False,
            use_cuda_graph=True,
            grad_iters=None,
            regularization=False,
        )
        self.left_ik_solver = IKSolver(self.left_ik_config)
        self.right_ik_solver = IKSolver(self.right_ik_config)
        self.last_shape = None
        self.last_dtype = None

    def solve(
        self,
        left_wrist_transform,
        right_wrist_transform,
        joint_4_to_10,
        joint_13_to_19,
        lift,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        if (
            left_wrist_transform.shape != self.last_shape
            or left_wrist_transform.dtype != self.last_dtype
        ):
            self.last_shape = left_wrist_transform.shape
            self.last_dtype = left_wrist_transform.dtype
            infer_mode = False
        else:
            infer_mode = True
        with torch.inference_mode(infer_mode):
            left_pose = Pose(
                position=left_wrist_transform[..., :3, 3] - torch.tensor([[0.0, 0.0, lift]], device=left_wrist_transform.device),
                rotation=left_wrist_transform[..., :3, :3],
            )
            right_pose = Pose(
                position=right_wrist_transform[..., :3, 3] - torch.tensor([[0.0, 0.0, lift]], device=left_wrist_transform.device),
                rotation=right_wrist_transform[..., :3, :3],
            )
            joints = torch.cat([lift, joint_4_to_10, joint_13_to_19], dim=-1)
            res_l = self.left_ik_solver.solve_batch(
                left_pose,
                retract_config=joint_4_to_10,
                seed_config=joint_4_to_10[None],
                return_seeds=1,
            )
            res_r = self.right_ik_solver.solve_batch(
                right_pose,
                retract_config=joint_13_to_19,
                seed_config=joint_13_to_19[None],
                return_seeds=1,
            )
        return (
            res_l.solution[:, 0, :],
            res_r.solution[:, 0, :],
        )

    def solve_single_arm(
        self,
        wrist_transform: torch.Tensor,
        joints: torch.Tensor,
        lift: torch.Tensor,
        left=True,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """
        wrist_transform: (num_env x 4 x 4)
        joints: (num_env x 7)
        lift: (num_env x 1)
        """
        if (
            wrist_transform.shape != self.last_shape
            or wrist_transform.dtype != self.last_dtype
        ):
            # this check is now not enough, use after checking it.
            self.last_shape = wrist_transform.shape
            self.last_dtype = wrist_transform.dtype
            infer_mode = False
        else:
            infer_mode = True
        with torch.inference_mode(infer_mode):
            pose = Pose(
                position=wrist_transform[..., :3, 3],
                rotation=wrist_transform[..., :3, :3],
            )
            joints_curobo = torch.cat([lift, joints], dim=-1)
            if left:
                res = self.left_ik_solver.solve_batch(
                    pose,
                    retract_config=joints_curobo,
                    seed_config=joints_curobo[None],
                    return_seeds=1,
                )
            else:
                res = self.right_ik_solver.solve_batch(
                    pose,
                    retract_config=joints_curobo,
                    seed_config=joints_curobo[None],
                    return_seeds=1,
                )
            q_solution: torch.Tensor = res.solution
        return q_solution

class UR5PinkIK_Pink:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: list,
        init_q: np.ndarray = None,  # 允许传入初始关节位置
        orientation_cost: float = 1.0,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        joint_names_to_lock: List[str] = [],
        ee_link_name: str = "",
    ):
        # 加载完整机器人模型
        self.full_robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=package_dirs,
            root_joint=None,
        )

        # 保存关节信息
        self.joint_names = [name for name in self.full_robot.model.names]
        self.active_joint_ids = []
        self.locked_joint_ids = []

        # 确定活动关节和锁定关节
        for joint_id in range(1, self.full_robot.model.njoints):  # 跳过universe关节
            joint_name = self.full_robot.model.names[joint_id]
            if joint_name in joint_names_to_lock:
                self.locked_joint_ids.append(joint_id)
            else:
                self.active_joint_ids.append(joint_id)

        # 保存完整机器人的默认配置
        self.full_q0 = np.copy(self.full_robot.q0)
        self.full_q0[0] = -1.57
        self.full_q0[1] = -1.57

        # 如果提供了初始姿态，使用它来更新默认配置
        if init_q is not None:
            assert len(init_q) == len(
                self.full_q0
            ), f"初始位置长度不匹配: {len(init_q)} vs {len(self.full_q0)}"
            self.full_q0 = np.copy(init_q)

        # 构建精简模型
        self.robot = self.full_robot.buildReducedRobot(self.locked_joint_ids)

        # 映射从完整机器人到精简机器人的关节位置
        self.reduced_indices = []
        reduced_joint_names = [
            self.robot.model.names[i] for i in range(self.robot.model.njoints)
        ]
        for joint_id in self.active_joint_ids:
            joint_name = self.full_robot.model.names[joint_id]
            if joint_name in reduced_joint_names:
                self.reduced_indices.append(reduced_joint_names.index(joint_name))

        # 创建适用于精简模型的初始配置
        self.reduced_q0 = np.copy(self.robot.q0)

        # 从完整配置更新精简配置中的活动关节
        self._update_reduced_q_from_full(self.full_q0)

        # 初始化配置
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

        # 初始化任务
        self.ee_task = FrameTask(
            ee_link_name,  # 需根据实际URDF调整
            position_cost=1.0,
            orientation_cost=orientation_cost,
        )
        self.posture_task = PostureTask(cost=1e-3)

        # 从当前配置设置姿态任务目标
        self.posture_task.set_target_from_configuration(self.configuration)

        self.tasks = [
            self.ee_task,
            self.posture_task,
        ]

        self.ee_link_name = ee_link_name

        # QP求解器配置
        self.solver = qpsolvers.available_solvers[0]
        self.dt = 0.01
        self.damp = damp

        # 初始化滤波器
        if use_in_lp_filter:
            self.filter = LPConstrainedSE3Filter(in_lp_alpha)
        else:
            self.filter = None

        # 输出关节信息
        print(f"活动关节数量: {len(self.active_joint_ids)}")
        print(
            f"活动关节: {[self.full_robot.model.names[i] for i in self.active_joint_ids]}"
        )

    def _update_reduced_q_from_full(self, full_q):
        """从完整配置更新精简模型配置"""
        # 遍历活动关节，将值从完整配置复制到精简配置
        for i, joint_id in enumerate(self.active_joint_ids):
            joint_name = self.full_robot.model.names[joint_id]
            joint_idx = self.full_robot.model.getJointId(joint_name)

            # 获取关节在完整模型配置中的索引范围
            joint = self.full_robot.model.joints[joint_idx]
            q_idx = self.full_robot.model.idx_qs[joint_idx]
            nq = joint.nq

            # 获取关节在精简模型中的索引
            reduced_idx = self.robot.model.getJointId(joint_name)
            reduced_q_idx = self.robot.model.idx_qs[reduced_idx]

            # 复制配置
            self.reduced_q0[reduced_q_idx : reduced_q_idx + nq] = full_q[
                q_idx : q_idx + nq
            ]

    def set_init_position(self, init_q):
        """设置新的初始位置"""
        assert len(init_q) == len(
            self.full_q0
        ), f"初始位置长度不匹配: {len(init_q)} vs {len(self.full_q0)}"
        self.full_q0 = np.copy(init_q)

        # 更新精简模型的初始配置
        self._update_reduced_q_from_full(self.full_q0)

        # 更新当前配置
        self.configuration.q = np.copy(self.reduced_q0)

        # 重置任务目标
        self.posture_task.set_target_from_configuration(self.configuration)

        # 重置滤波器
        if self.filter is not None:
            self.filter.reset()

        return self.configuration.q

    def solve(
        self,
        wrist_transform,
        joints,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """求解双臂逆运动学

        Args:
            wrist_transform: 4x4 齐次变换矩阵，表示ee目标位姿
            orientation_cost: 可选方向权重覆盖值

        Returns:
            (关节位置)
        """
        # 更新方向权重
        device = "cuda"
        if isinstance(joints, torch.Tensor):
            device = joints.device
            joints = joints.cpu().numpy()
        if isinstance(wrist_transform, torch.Tensor):
            device = wrist_transform.device
            wrist_transform = wrist_transform.cpu().numpy()
        if orientation_cost is not None:
            self.ee_task.orientation_cost = orientation_cost

        # 应用平滑滤波
        if self.filter is not None:
            transform = self.filter.next(wrist_transform)
        else:
            transform = wrist_transform

        self.reduced_q0 = joints
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.reduced_q0
        )

        # 设置任务目标
        self.ee_task.set_target(pin.SE3(transform))

        initial_error = np.linalg.norm(
            self.ee_task.compute_error(self.configuration)
        ) + np.linalg.norm(self.ee_task.compute_error(self.configuration))
        error_norm = initial_error

        logger.trace(f"Initial error norm: {error_norm}")

        nb_steps = 0
        damping_factor = 1e-8  #  1e-3
        prev_error = float("inf")
        current_dt = dt
        try:
            while error_norm > stop_thres and (
                nb_steps < max_try_times
                or (nb_steps < max_try_times and prev_error - error_norm > 0.003)
            ):
                dv = solve_ik(
                    self.configuration,
                    tasks=self.tasks,
                    dt=current_dt,
                    damping=damping_factor,
                    safety_break=False,
                    solver="quadprog",
                )
                if nb_steps % 5 == 0:
                    damping_factor *= 1.5
                # 如果误差减小不明显，减小步长
                if prev_error - error_norm < error_norm * 0.01:
                    current_dt *= 0.8
                prev_error = error_norm
                q_out = pin.integrate(
                    self.robot.model, self.configuration.q, dv * current_dt
                )
                self.configuration = pink.Configuration(
                    self.robot.model, self.robot.data, q_out
                )
                pin.updateFramePlacements(self.robot.model, self.robot.data)
                error_norm = np.linalg.norm(
                    self.ee_task.compute_error(self.configuration)
                )
                nb_steps += 1
        except Exception as e:
            logger.error(f"IK failed: {e}")
            return torch.from_numpy(self.configuration.q).to(device)

        # print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
        return torch.from_numpy(self.configuration.q).to(device)


class UR5PinkIK_Curobo:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: list,
        init_q: np.ndarray = None,
        orientation_cost: float = 1.0,
        damp: float = 1e-6,
        use_in_lp_filter: bool = True,
        in_lp_alpha: float = 0.9,
        joint_names_to_lock: List[str] = [],
        ee_link_name: str = "",
        base_link_name: str = "",
        use_cuda_graph=True,
    ):
        """初始化 UR5 IK 求解器"""
        # Essential parameters
        self.ee_link_name = ee_link_name if ee_link_name else "wrist_3_link"
        self.base_link_name = base_link_name if base_link_name else "base_link"
        self.orientation_cost = orientation_cost
        self.use_in_lp_filter = use_in_lp_filter
        self.in_lp_alpha = in_lp_alpha
        self.filter_state = None

        with torch.inference_mode(False):
            self.tensor_args = TensorDeviceType(device=torch.device("cuda"))
            # Create robot config and IK solver
            self.robot_cfg = RobotConfig.from_basic(
                urdf_path, self.base_link_name, self.ee_link_name, self.tensor_args
            )

            # Setup default configuration
            self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)

            self.dof = self.kin_model.get_dof()
            self.full_q0 = np.zeros(self.dof)
            if self.dof >= 6:  # Assuming standard UR5 joint layout
                self.full_q0[0] = -1.57
                self.full_q0[1] = -1.57

            # Use provided initial configuration if available
            if init_q is not None:
                assert len(init_q) == len(
                    self.full_q0
                ), f"Initial position length mismatch"
                self.full_q0 = np.copy(init_q)

            # Configure and initialize IK solver
            self.ik_config = IKSolverConfig.load_from_robot_config(
                self.robot_cfg,
                None,
                position_threshold=0.005,
                num_seeds=1,
                self_collision_check=True,
                self_collision_opt=True,
                tensor_args=self.tensor_args,
                high_precision=False,
                use_cuda_graph=use_cuda_graph,
                grad_iters=None,
                regularization=True,
            )
            self.ik_solver = IKSolver(self.ik_config)

        self.last_shape = None
        self.last_dtype = None

    def solve_batch(
        self,
        target_pos,
        target_quat,
        joints_batch,
        orientation_cost=None,
        stop_thres=0.02,
        dt=0.01,
        max_try_times=20,
    ):
        """批量IK问题并行求解"""
        # Cuda_graph is required to be generated when the first time to solve IK so the inference mode must be False. And afterwards solve will update tensor in place, so the inference mode must be True
        if target_pos.shape != self.last_shape or target_pos.dtype != self.last_dtype:
            self.last_shape = target_pos.shape
            self.last_dtype = target_pos.dtype
            self.ik_solver.reset_cuda_graph()
        #     infer_mode = False
        # else:
        #     infer_mode = True
        with torch.inference_mode(False):
            target_pos = target_pos.clone()
            target_quat = target_quat.clone()
            joints_batch = joints_batch.clone()
            pose = Pose(target_pos, target_quat)
            result = self.ik_solver.solve_batch(
                pose,
                retract_config=joints_batch,
                seed_config=joints_batch[None],
                return_seeds=1,
            )

            if (
                not result.success.all()
            ):  # This is to solve a weired bug that the ik solver return false at some points when the cuda graph cached something, which will be solvable if we reset the graph
                result = self.ik_solver.solve_batch(
                    pose,
                    retract_config=joints_batch,
                    seed_config=joints_batch[None],
                    return_seeds=1,
                    warm_up=True,
                )
        q_solution: torch.Tensor = result.solution
        return q_solution[:, 0, :]


if __name__ == "__main__":
    robot = X7PinkIK(
        use_in_lp_filter=False,
    )
    import roboticstoolbox as rtb

    rtb_robot = rtb.Robot.URDF(f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf")
    full_robot = robot.full_robot

    for i in range(100):
        q0 = full_robot.q0 + np.random.randn(full_robot.model.nq) * 0.1
        full_robot.framesForwardKinematics(q0)
        rtb_ee = rtb_robot.fkine(q0, end="link10").A
        pin_ee = np.eye(4)
        pin_ee[:3, 3] = full_robot.data.oMf[
            full_robot.model.getFrameId("link10")
        ].translation
        pin_ee[:3, :3] = full_robot.data.oMf[
            full_robot.model.getFrameId("link10")
        ].rotation
        assert np.allclose(rtb_ee, pin_ee, rtol=1e-3, atol=1e-3)

    print(1)
