# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
import time

import numpy as np

try:
    from oculus_reader.reader import OculusReader
except:
    print("OculusReader not found, teleop will not work")

# Use environment variable for Isaac Sim path or skip if not available
import os

isaac_sim_franka_path = os.environ.get("ISAAC_SIM_FRANKA_PATH")
if isaac_sim_franka_path:
    sys.path.append(isaac_sim_franka_path)
else:
    # Try common Isaac Sim installation paths
    possible_paths = [
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.0.0/src/franka"),
        "/opt/nvidia/isaac-sim/src/franka",
        "/usr/local/isaac-sim/src/franka",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.append(path)
            break

import threading
from copy import deepcopy
from multiprocessing import Lock
from pdb import set_trace

from scipy.spatial.transform import Rotation as R


def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()

    return thread


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


def get_transformation(pos, rmat):
    transformation = np.eye(4)
    transformation[:3, :3] = rmat
    transformation[:3, 3] = pos
    return transformation


class DualArmVRPolicySingle:
    def __init__(
        self,
        right_controller: bool = True,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
        rmat_reorder: list = [-2, -1, -3, 4],
    ):
        self.vr_to_global_mat = np.eye(4)
        self.robot_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self._state_lock = Lock()
        self._last_time_enable_rot = False

        # # x forward, y left, z up
        # self._transform_mat = np.zeros((4, 4))
        # self._transform_mat[3, 3] = 1
        # self._transform_mat[0, 1] = -1
        # self._transform_mat[1, 0] = -1
        # self._transform_mat[2, 2] = -1

        # x forward, y left, z up
        # 30 degree
        self._transform_mat = np.array(
            [
                [0, -np.sqrt(3) / 2, 1 / 2, 0],
                [-1, 0, 0, 0],
                [0, -1 / 2, -np.sqrt(3) / 2, 0],
                [0, 0, 0, 1],
            ]
        )

        self._tmp_transform_mat = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
        )

        # 60 degree
        self._transform_mat = self._tmp_transform_mat @ np.array(
            [
                [0, -1 / 2, np.sqrt(3) / 2, 0],
                [-1, 0, 0, 0],
                [0, -np.sqrt(3) / 2, -1 / 2, 0],
                [0, 0, 0, 1],
            ]
        )

        self.reset_state()

        # controlling gripper

    def reset_state(self):
        self._state = {
            "movement_enabled": False,
            "poses": {},
            "buttons": {},
        }
        self.reset_origin = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None

    def _update_internal_state(
        self, transformations_and_buttons, last_read_time, num_wait_sec=5, hz=50
    ):
        # Read Controller
        # print("reading controller")
        poses, buttons = transformations_and_buttons
        # print("poses :" , poses)
        if poses == {}:
            with self._state_lock:
                self._state["poses"] = {}
            return

        # Determine Control Pipeline #
        # toggled = self._state["movement_enabled"] != buttons[f"{self.controller_id.upper()}G"]
        # self.update_sensor = self.update_sensor or buttons[f"{self.controller_id.upper()}G"]
        # self.reset_orientation = self.reset_orientation or buttons[f"{self.controller_id.upper()}J"]
        # self.reset_origin = self.reset_origin or toggled

        # Save Info #
        with self._state_lock:
            self._state["poses"] = poses
            self._state["buttons"] = buttons

    # def _process_reading(self):
    #     rot_mat = np.asarray(self._state["poses"][self.controller_id])
    #     # rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
    #     vr_pos = self.spatial_coeff * rot_mat[:3, 3]
    #     vr_quat = rmat_to_quat(rot_mat[:3, :3])

    #     if self.vr_state is None:
    #         self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": 1.0}
    #     else:
    #         self.vr_state["pos"] = vr_pos
    #         self.vr_state["quat"] = vr_quat

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(
        self, poses, include_info=False, action_type="cartesian_pose"
    ):
        # Read Sensor #=
        # Read Observation
        robot_pos = poses["translation"]
        robot_rmat = poses["rotation"]
        with self._state_lock:
            gripper = 0.044 * (
                1 - self._state["buttons"][f"{self.controller_id.upper()}Tr"]
            )

            if (self._state["buttons"][f"A"] and self.controller_id == "r") or (
                self._state["buttons"][f"X"] and self.controller_id == "l"
            ):
                return {
                    "position": robot_pos,
                    "rmat": robot_rmat,
                    "gripper": gripper,
                    "reset": True,
                }

            # self._process_reading()
            if (
                not self._last_time_enable_rot
                and self._state["buttons"][f"{self.controller_id.upper()}G"]
            ):
                self.vr_to_global_mat = self._state["poses"][self.controller_id]
                self.robot_to_global_mat = get_transformation(robot_pos, robot_rmat)

            self._last_time_enable_rot = self._state["buttons"][
                f"{self.controller_id.upper()}G"
            ]

            if not self._last_time_enable_rot:
                return {
                    "position": robot_pos,
                    "rmat": robot_rmat,
                    "gripper": gripper,
                    "reset": False,
                }

            delta_tranform = (
                np.linalg.inv(self.vr_to_global_mat)
                @ self._state["poses"][self.controller_id]
            )
            delta_tranform = (
                self._transform_mat
                @ delta_tranform
                @ np.linalg.inv(self._transform_mat)
            )
            robot_transform = get_transformation(robot_pos, robot_rmat)
            robot_transform_new = self.robot_to_global_mat @ delta_tranform
            print("delta_tranform[:3, 3]: ", delta_tranform[:3, 3])
            robot_rmat_new = robot_transform_new[:3, :3]
            robot_pos_new = robot_transform_new[:3, 3]

            return {
                "position": robot_pos_new,
                "rmat": robot_rmat_new,
                "gripper": gripper,
                "reset": False,
            }

    def get_info(self):
        return {
            "success": self._state["buttons"]["A"],
            "failure": self._state["buttons"]["B"],
            "movement_enabled": self._state["movement_enabled"],
        }


class DualArmVRPolicy:
    def __init__(
        self,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
        rmat_reorder: list = [-2, -1, -3, 4],
    ):
        self.right_controller = DualArmVRPolicySingle(
            right_controller=True,
            max_lin_vel=max_lin_vel,
            max_rot_vel=max_rot_vel,
            max_gripper_vel=max_gripper_vel,
            spatial_coeff=spatial_coeff,
            pos_action_gain=pos_action_gain,
            rot_action_gain=rot_action_gain,
            gripper_action_gain=gripper_action_gain,
            rmat_reorder=rmat_reorder,
        )
        self.left_controller = DualArmVRPolicySingle(
            right_controller=False,
            max_lin_vel=max_lin_vel,
            max_rot_vel=max_rot_vel,
            max_gripper_vel=max_gripper_vel,
            spatial_coeff=spatial_coeff,
            pos_action_gain=pos_action_gain,
            rot_action_gain=rot_action_gain,
            gripper_action_gain=gripper_action_gain,
            rmat_reorder=rmat_reorder,
        )
        self.oculus_reader = OculusReader()
        self._state_lock = Lock()
        self._state = {
            "poses": {},
            "buttons": {},
        }

        run_threaded_command(self._update_internal_state)

    def reset_state(self):
        self.right_controller.reset_state()
        self.left_controller.reset_state()

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            time.sleep(1 / hz)
            transformations_and_buttons = (
                self.oculus_reader.get_transformations_and_buttons()
            )
            with self._state_lock:
                self._state["poses"] = transformations_and_buttons[0]
                self._state["buttons"] = transformations_and_buttons[1]
            self.right_controller._update_internal_state(
                transformations_and_buttons,
                last_read_time,
                num_wait_sec=num_wait_sec,
                hz=hz,
            )
            self.left_controller._update_internal_state(
                transformations_and_buttons,
                last_read_time,
                num_wait_sec=num_wait_sec,
                hz=hz,
            )

    def _calculate_action(self, poses, include_info=False):
        actions = {}
        actions["r"] = self.right_controller._calculate_action(
            poses["r"], include_info=include_info
        )
        actions["l"] = self.left_controller._calculate_action(
            poses["l"], include_info=include_info
        )
        with self._state_lock:
            actions["lift_speed"] = self._state["buttons"]["rightJS"][1]
            actions["yaw_speed"] = -self._state["buttons"]["rightJS"][0]
            actions["forward_speed"] = self._state["buttons"]["leftJS"][1]
            actions["side_speed"] = -self._state["buttons"]["leftJS"][0]
        return actions

    def forward(self, poses, include_info=False):
        if (
            self.left_controller._state["poses"] == {}
            or self.right_controller._state["poses"] == {}
        ):
            action = {"r": None, "l": None}
            if include_info:
                return action, {}
            else:
                return action
        return self._calculate_action(poses, include_info=include_info)
