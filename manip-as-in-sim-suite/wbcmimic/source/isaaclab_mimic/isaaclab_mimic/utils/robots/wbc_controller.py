# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import qpsolvers as qp
import roboticstoolbox as rtb
from loguru import logger as lgr


@dataclass
class WbcControllerCfg:
    dt: float = 0.01
    urdf: str = ""
    ee: str = ""
    init_q: np.ndarray = None
    active_joint_idx: List = None
    velocity_idx: List = None
    threshold: float = 0.05
    wbc_type: str = "omni"
    selected_joints: List[str] = None
    p_servo_gain: float = 4
    v_limit: float = (
        0.5  # Maximum joint velocity limit, used for omni and two-wheel wbc types
    )
    max_v_limit: float = 0.5
    """ choose from omni, two-wheel, none """


class WbcController:
    def __init__(self, cfg: WbcControllerCfg):
        self.robot = rtb.Robot.URDF(cfg.urdf)
        self.dt = cfg.dt
        self.goal = None
        self.cfg = cfg
        if cfg.init_q is not None:
            self.update_init_pos(cfg.init_q)
        if self.cfg.active_joint_idx is None:
            self.cfg.active_joint_idx = list(range(self.robot.n))
            self.fake_robot_dim = self.robot.n
        else:
            self.fake_robot_dim = len(self.cfg.active_joint_idx)
        if self.cfg.velocity_idx is None:
            self.cfg.velocity_idx = [0, 1, 2]
        if self.cfg.wbc_type == "omni":
            self.wbc_dim = 3
        elif self.cfg.wbc_type == "two-wheel":
            self.wbc_dim = 2
        elif self.cfg.wbc_type == "none":
            self.wbc_dim = 0
        else:
            raise ValueError("wbc_type must be omni, two-wheel or none")

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    def update_joint_pos(self, init_q: np.ndarray):
        self.robot.q[self.cfg.active_joint_idx] = init_q

    def fkine(self, q: np.ndarray, end: str = None):
        if end is None:
            end = self.cfg.ee
        tmp_q = np.copy(self.robot.q)
        tmp_q[self.cfg.active_joint_idx] = q
        return self.robot.fkine(tmp_q, end=end).A

    def update_root_pose(self, pose: np.ndarray):
        self.robot._T = pose

    def step_robot(self, vel_multiplier: float = 1.0):
        assert self.goal is not None, "Goal has not been set, plz set goal first."

        wTe = self.robot.fkine(self.robot.q, end=self.cfg.ee)

        eTep = np.linalg.inv(wTe) @ self.goal

        # Spatial error
        et = np.sum(np.abs(eTep[:3, -1]))

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(self.fake_robot_dim + 6)

        # Joint velocity component of Q
        Q[: self.fake_robot_dim, : self.fake_robot_dim] *= Y
        Q[: self.wbc_dim, : self.wbc_dim] *= 1.0 / et

        # Slack component of Q
        Q[self.fake_robot_dim :, self.fake_robot_dim :] = (1.0 / et) * np.eye(6)

        v, _ = rtb.p_servo(wTe, self.goal, self.cfg.p_servo_gain * vel_multiplier)

        v[self.wbc_dim :] *= 1.3

        local_lim = min(self.cfg.v_limit * vel_multiplier, self.cfg.max_v_limit)

        v[self.wbc_dim : self.wbc_dim + 3] = np.clip(
            v[self.wbc_dim : self.wbc_dim + 3], -local_lim, local_lim
        )

        # The equality contraints
        Aeq = np.c_[self.robot.jacobe(self.robot.q, end=self.cfg.ee), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((self.fake_robot_dim + 6, self.fake_robot_dim + 6))
        bin = np.zeros(self.fake_robot_dim + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.1

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
        c = np.concatenate(
            (
                np.zeros(self.wbc_dim),
                -self.robot.jacobm(
                    start=self.robot.links[self.wbc_dim], end=self.cfg.ee
                ).reshape((self.fake_robot_dim - self.wbc_dim,)),
                np.zeros(6),
            )
        )

        # Get base to face end-effector
        kε = 0.5
        bTe = self.robot.fkine(self.robot.q, include_base=False, end=self.cfg.ee).A
        θε = math.atan2(bTe[1, -1], bTe[0, -1])
        ε = kε * θε
        if self.wbc_dim != 0:
            c[0] = -ε

        # The lower and upper bounds on the joint velocity and slack variable
        lb = np.r_[self.robot.qlim[0, self.cfg.active_joint_idx], -10 * np.ones(6)]
        ub = np.r_[self.robot.qlim[1, self.cfg.active_joint_idx], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
        if qd is None:
            lgr.warning("QP solution failed")
            qd = np.zeros_like(self.robot.q)
        qd = qd[: self.fake_robot_dim]
        if et > 0.5:
            qd *= 0.7 / et
        else:
            qd *= 1.4
        qd_all = np.zeros_like(self.robot.q)
        qd_all[self.cfg.active_joint_idx] = qd
        import copy

        self._last_q = copy.deepcopy(self.robot.q)
        self.robot.q = self.robot.q + qd_all * self.dt
        self._last_T = copy.deepcopy(self.robot._T)
        self.robot._T = self.robot.fkine(
            self.robot._q, end=self.robot.links[self.wbc_dim]
        ).A
        self.robot.q[: self.wbc_dim] = 0

        if et < self.cfg.threshold:
            return True, qd_all[self.cfg.active_joint_idx]
        else:
            return False, qd_all[self.cfg.active_joint_idx]


if __name__ == "__main__":
    import spatialgeometry as sg

    # Get URDF path from environment variable or use relative path
    import os

    ur5_urdf_path = os.environ.get(
        "UR5_URDF_PATH",
        os.path.join(
            os.path.dirname(__file__),
            "../../../../../assets/UR5/ur5_isaac_simulation/robot.urdf",
        ),
    )
    robot2 = rtb.Robot.URDF(ur5_urdf_path)

    ax_goal = sg.Axes(0.1)
    robot2.q = np.array(
        [-3.02692452, -2.00677957, -1.50796447, -1.12242124, 1.59191481, -0.055676]
        + [0.0] * 6
    )
    print(robot2.q)
    goal = robot2.fkine(robot2.q, end="wrist_3_link")
    goal.A[:2, 3] += np.random.randn(2) * 0.2
    # goal.A[2, 3] += 0.05
    ax_goal.T = goal
    dt = 0.025
    controller_cfg = WbcControllerCfg(
        dt=0.15,
        urdf=ur5_urdf_path,
        ee="wrist_3_link",
        active_joint_idx=[0, 1, 2, 3, 4, 5],
        threshold=0.02,
    )
    controller = WbcController(controller_cfg)
    controller.update_joint_pos(robot2.q)
    controller.set_goal(goal.A)
    arrived = False
    robot2.q = controller.robot.q
    while not arrived:
        arrived, qd = controller.step_robot()
    print("Done")
