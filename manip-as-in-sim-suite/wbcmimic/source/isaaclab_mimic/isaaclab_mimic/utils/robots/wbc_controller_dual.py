# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import math
from collections import OrderedDict
from dataclasses import MISSING, dataclass
from typing import Dict, List

import numpy as np
import qpsolvers as qp
import roboticstoolbox as rtb
import torch
from loguru import logger as lgr
from scipy.spatial.transform import Rotation as R

if __name__ != "__main__":
    from ..exceptions import (
        CollisionError,
        WbcSolveFailedException,
        WbcStepOverMaxStepException,
    )


def safe_np(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x


@dataclass
class DualArmWbcControllerCfg:
    dt: float = 0.01
    urdf: str = ""
    ees: List[str] = None
    init_q: np.ndarray = None
    active_joint_idx: List = None
    velocity_idx: List = None
    threshold: float = 0.02
    d_stop = 0.03
    d_damp = 0.08


class DualArmWbcController:
    def __init__(self, cfg: DualArmWbcControllerCfg):
        self.robot = rtb.Robot.URDF(cfg.urdf)
        self.dt = cfg.dt
        self.goals: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cfg = cfg
        if cfg.init_q is not None:
            self.update_joint_pos(cfg.init_q)
        if self.cfg.active_joint_idx is None:
            self.cfg.active_joint_idx = list(range(self.robot.n))
            self.fake_robot_dim = self.robot.n
        else:
            self.fake_robot_dim = len(self.cfg.active_joint_idx)
        if self.cfg.velocity_idx is None:
            self.cfg.velocity_idx = [0, 1, 2]

        self.ee_to_fake_robot_index = {}
        active_joint_idx_np = np.array(self.cfg.active_joint_idx)
        self.index_contrib_nums = np.zeros((self.fake_robot_dim + 6), dtype=np.float32)
        for ee in self.cfg.ees:
            ee_joint_idx_np = self.get_related_joint_idx(ee)
            mask = np.isin(active_joint_idx_np, ee_joint_idx_np)
            self.ee_to_fake_robot_index[ee] = np.where(mask)[0]
            self.index_contrib_nums[self.ee_to_fake_robot_index[ee]] += 1
        self.index_contrib_nums[self.index_contrib_nums == 0] = 1
        self.d_stop = self.cfg.d_stop
        self.d_damp = self.cfg.d_damp

    def get_related_joint_idx(self, ee_name: str):
        return self.robot.ets(self.robot.base_link.name, ee_name).jindices

    def set_goal(self, ee: str, goal: np.ndarray):
        self.goals[ee] = goal

    def update_joint_pos(self, init_q: np.ndarray):
        self.robot.q = init_q

    def update_root_pose(self, pose: np.ndarray):
        self.robot._T = pose

    def step_robot(self, distance=None, norm_a2b=None, index_a=None, body_names=None):
        assert self.goals, "Goal has not been set, plz set goal first."

        if distance is None:
            distance = []
            norm_a2b = []
            index_a = []
        else:
            try:
                assert norm_a2b is not None, "norm_a2b is None"
                assert index_a is not None, "index_a is None"
                assert len(distance) == len(norm_a2b), "len(distance)!= len(norm_a2b)"
                assert len(distance) == len(index_a), "len(distance)!= len(index_a)"
            except:
                distance = []
                norm_a2b = []
                index_a = []

        distance = safe_np(distance)
        norm_a2b = safe_np(norm_a2b)
        index_a = safe_np(index_a)

        wTes = OrderedDict()
        et = 0.0
        for ee in self.cfg.ees:
            wTe = self.robot.fkine(self.robot.q, end=ee).A
            eTep = np.linalg.inv(wTe) @ self.goals[ee]
            # Spatial error
            et += np.sum(np.abs(eTep[:3, -1]))

            wTes[ee] = wTe
        et = et / len(self.cfg.ees)
        # Gain term (lambda) for control minimisation
        Y = 0.01

        # 修改：为每个末端执行器分配独立的松弛变量
        num_slack_vars = 6 * len(self.cfg.ees)  # 每个末端执行器6个松弛变量
        total_vars = self.fake_robot_dim + num_slack_vars

        # Quadratic component of objective function
        Q = np.eye(total_vars)

        # Joint velocity component of Q
        Q[: self.fake_robot_dim, : self.fake_robot_dim] *= Y
        et_safe = max(et, 1e-3)  # 防止除零
        Q[:3, :3] *= min(1.0 / (et_safe**4), 10)  # 限制最大权重

        # 松弛变量权重：每个末端执行器的松弛变量都有相同的权重
        slack_weight = min(1.0 / (et_safe**4), 10)
        for i in range(len(self.cfg.ees)):
            start_idx = self.fake_robot_dim + i * 6
            end_idx = self.fake_robot_dim + (i + 1) * 6
            Q[start_idx:end_idx, start_idx:end_idx] = slack_weight * np.eye(6)
        # Slack component of Q
        # if et > 0.1:
        #     Q[self.fake_robot_dim :, self.fake_robot_dim :] = (1.0 / (et)) * np.eye(6)
        # else:

        Aeqs = {}
        beqs = {}
        for i, ee in enumerate(self.cfg.ees):
            gain = 7 if et < 0.1 else 3
            # gain = 1.5
            v, _ = rtb.p_servo(wTes[ee], self.goals[ee], gain)

            v[3:] *= 1.3

            Aeq_each = np.zeros((6, total_vars), dtype=np.float64)
            # The equality contraints
            Aeq_each[:, self.ee_to_fake_robot_index[ee]] = np.c_[
                self.robot.jacobe(self.robot.q, end=ee)
            ]

            # 每个末端执行器使用自己的松弛变量
            slack_start = self.fake_robot_dim + i * 6
            slack_end = self.fake_robot_dim + (i + 1) * 6
            Aeq_each[:, slack_start:slack_end] = np.eye(6)

            beq_each = v.reshape((6,))
            Aeqs[ee] = Aeq_each
            beqs[ee] = beq_each

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((self.fake_robot_dim + num_slack_vars, total_vars))
        bin = np.zeros(self.fake_robot_dim + num_slack_vars)

        # Add collision avoidance constraints
        if len(distance) > 0:
            # Create temporary matrices for collision constraints
            num_collision_pairs = len(distance)
            Ain_collision = np.zeros((num_collision_pairs, total_vars))
            bin_collision = np.zeros(num_collision_pairs)

            for i in range(num_collision_pairs):
                # Calculate Jacobian for the collision point
                J = np.zeros((3, len(self.robot.q)))
                J[:, self.get_related_joint_idx(body_names[index_a[i]])] = (
                    self.robot.jacob0(self.robot.q, end=body_names[index_a[i]])[:3, :]
                )

                # Project Jacobian along collision normal direction
                J_collision = norm_a2b[i : i + 1] @ J[:3, :]

                # Add to inequality constraints
                Ain_collision[i, : self.fake_robot_dim] = J_collision[
                    0, self.cfg.active_joint_idx
                ]
                bin_collision[i] = (distance[i] - self.d_stop) / (
                    self.d_damp - self.d_stop
                )

            # Combine with existing inequality constraints
            Ain = np.vstack([Ain, Ain_collision])
            bin = np.hstack([bin, bin_collision])

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.01

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        tmp_1, tmp_2 = self.robot.joint_velocity_damper(ps, pi)
        Ain[: self.fake_robot_dim, : self.fake_robot_dim] = tmp_1[
            :, self.cfg.active_joint_idx
        ][self.cfg.active_joint_idx, :]
        bin[: self.fake_robot_dim] = tmp_2[self.cfg.active_joint_idx]
        # Linear component of objective function: the manipulability Jacobian
        c = np.zeros((total_vars,), dtype=np.float64)

        for ee in self.cfg.ees:
            c[self.ee_to_fake_robot_index[ee]] += np.concatenate(
                (
                    np.zeros(3),
                    -self.robot.jacobm(start=self.robot.links[4], end=ee).reshape((-1)),
                )
            )

        # Get base to face end-effector
        kε = 0.5
        θεs = []
        for ee in self.cfg.ees:
            bTe = self.robot.fkine(self.robot.q, include_base=False, end=ee).A
            θεs.append(math.atan2(bTe[1, -1], bTe[0, -1]))
        θε = np.mean(θεs)
        ε = kε * θε
        c[2] = -ε

        # The lower and upper bounds on the joint velocity and slack variable
        lb = np.r_[
            self.robot.qlim[0, self.cfg.active_joint_idx],
            -1e6 * np.ones(num_slack_vars),
        ]
        ub = np.r_[
            self.robot.qlim[1, self.cfg.active_joint_idx], 1e6 * np.ones(num_slack_vars)
        ]
        # # Solve for the joint velocities dq
        # qd = np.zeros(self.fake_robot_dim + 6, dtype=np.float32)
        # for ee in self.cfg.ees:
        #     qd_ = qp.solve_qp(Q, c, Ain, bin, Aeqs[ee], beqs[ee], lb=lb, ub=ub, solver="quadprog")
        #     if qd_ is None:
        #         break
        #     qd[self.ee_to_fake_robot_index[ee]] += qd_[self.ee_to_fake_robot_index[ee]]

        # qd[: self.fake_robot_dim] = qd[: self.fake_robot_dim] / self.index_contrib_nums[: self.fake_robot_dim]

        Aeq = np.zeros((6 * len(self.cfg.ees), total_vars))
        beq = np.zeros((6 * len(self.cfg.ees),))
        for i, ee in enumerate(self.cfg.ees):
            Aeq[i * 6 : (i + 1) * 6, :] = Aeqs[ee]
            beq[i * 6 : (i + 1) * 6] = beqs[ee]
        Q[2, 2] *= 200  # not to rotate base too much

        qd_ = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="proxqp")

        max_try_times = 5
        tried_times = 0
        ps_tmp = ps
        while qd_ is None and tried_times < max_try_times:
            ps_tmp *= 0.1

            # Form the joint limit velocity damper
            tmp_1, tmp_2 = self.robot.joint_velocity_damper(ps_tmp, pi)
            Ain[: self.fake_robot_dim, : self.fake_robot_dim] = tmp_1[
                :, self.cfg.active_joint_idx
            ][self.cfg.active_joint_idx, :]
            bin[: self.fake_robot_dim] = tmp_2[self.cfg.active_joint_idx]
            qd_ = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="osqp")

        if qd_ is None:
            lgr.warning("Qp failed!")
            raise WbcSolveFailedException

        qd = qd_[: self.fake_robot_dim]

        # if et > 0.5:
        #     qd *= 0.7 / et
        # else:
        #     qd *= 1.4
        qd_all = np.zeros_like(self.robot.q)
        qd_all[self.cfg.active_joint_idx] = qd
        import copy

        self._last_q = copy.deepcopy(self.robot.q)
        self.robot.q = self.robot.q + qd_all * self.dt
        self._last_T = copy.deepcopy(self.robot._T)
        self.robot._T = self.robot.fkine(self.robot.q, end=self.robot.links[3]).A
        self.robot.q[:3] = 0

        if et < self.cfg.threshold:
            return True, qd_all
        else:
            return False, qd_all


if __name__ == "__main__":
    import spatialgeometry as sg
    import spatialmath as sm
    import swift

    # robot = rtb.models.FrankieOmni()
    # robot2 = rtb.models.FrankieOmni()
    # Get URDF path from environment variable or use relative path
    import os

    urdf_path = os.environ.get(
        "X7_URDF_PATH",
        os.path.join(
            os.path.dirname(__file__), "../../../../../assets/X7/urdf/X7_2.urdf"
        ),
    )
    robot2 = rtb.Robot.URDF(urdf_path)

    goal_r = robot2.fkine(robot2.q, end="link19")
    goal_l = robot2.fkine(robot2.q, end="link10")
    # goal.A[:3, :3] = np.diag([-1, 1, -1])
    # goal.A[0, -1] -= 4.0
    # goal.A[2, -1] -= 0.25
    a = np.random.randn(2) * 1.2
    goal_r.A[:] = np.array(
        [
            [
                -0.19714783132076263,
                0.9338343143463135,
                -0.2984730303287506,
                2.8546833992004395,
            ],
            [
                -0.769888699054718,
                -0.33595895767211914,
                -0.5425888299942017,
                -2.1076481342315674,
            ],
            [
                -0.6069627404212952,
                0.12282078713178635,
                0.7851822376251221,
                1.070731520652771,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # goal_l.A[2, 3] += 0.05
    goal_l.A[:] = np.array(
        [
            [
                -0.8148020505905151,
                0.136772021651268,
                0.5633747577667236,
                3.368746519088745,
            ],
            [
                -0.5469205379486084,
                0.140975683927536,
                -0.8252295255661011,
                -2.1710658073425293,
            ],
            [
                -0.19229039549827576,
                -0.9805199503898621,
                -0.040063925087451935,
                0.9428311586380005,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # goal_r.A[2, 3] += 0.05
    # goal_r.A[1, 3] += 0.05
    # goal_r.A[0, 3] += 0.05

    # goal_l.A[:2, 3] -= 30
    # goal_r.A[:2, 3] -= 30
    dt = 0.025
    print(goal_l)
    print(goal_r)
    controller_cfg = DualArmWbcControllerCfg(
        dt=dt,
        urdf=urdf_path,
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
    )
    controller = DualArmWbcController(controller_cfg)
    controller.set_goal("link10", goal_l.A)
    controller.set_goal("link19", goal_r.A)
    controller.update_root_pose(
        np.array(
            [
                [
                    0.2558494210243225,
                    0.9667166471481323,
                    -7.322867645598308e-08,
                    3.1361427307128906,
                ],
                [
                    -0.9667166471481323,
                    0.2558494210243225,
                    -1.1470073246755419e-07,
                    -1.135977029800415,
                ],
                [
                    -9.214760154918622e-08,
                    1.0013749118797932e-07,
                    1.0,
                    0.18000000715255737,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    controller.update_joint_pos(
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.011125938288421117,
                4.372832052013109e-08,
                8.216004516725661e-07,
                -0.007527597316103834,
                0.007412296761476781,
                -0.006342906720438033,
                -0.025309920299704355,
                -0.04553010018121106,
                0.010666695901311073,
                0.018685805562550586,
                8.3105767600955e-08,
                2.623150308078692e-10,
                0.03006011123265281,
                0.010896635315301769,
                0.005612767423637731,
                0.025308832344564842,
                0.016426725155610833,
                0.027057979325573907,
                -0.017375462241640025,
                0.0,
                5.845597783604717e-08,
            ]
        )
    )
    # env = swift.Swift()
    # env.launch(realtime=True)

    # env.add(controller.robot)
    # env.step()
    arrived = False
    robot2.q = controller.robot.q
    while not arrived:
        arrived, qd = controller.step_robot()
        # robot2.q = robot2.q + qd * dt
        # robot_base_transform = robot2.fkine(robot2._q, end=robot2.links[3]).A
        # robot_base_transform[:2, 3] += np.random.randn(2) * qd[0] * 0.9
        # robot2._T = robot_base_transform
        # robot2.q[:2] = 0
        # controller.update_init_pos(robot2.q)
        # env.step(dt)
    print("Done")
