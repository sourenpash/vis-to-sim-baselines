# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Sequence, Tuple

import isaaclab.utils.math as PoseUtils
import numpy as np
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_mimic.envs import ManagerBasedRLMimicEnv
from isaaclab_mimic.utils.exceptions import WbcSolveFailedException
from isaaclab_mimic.utils.robots.wbc_controller import WbcController

from .ur5_joint_mimic_env import UR5BaseMimicJointControlEnv


class UR5JointMimicMPEnv(UR5BaseMimicJointControlEnv):
    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        ManagerBasedRLMimicEnv.__init__(self, cfg, render_mode, **kwargs)
        self.robot_entity_cfg = SceneEntityCfg(
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
        )
        self.robot_entity_cfg.resolve(self.scene)
        self.q0s = None

    def setup_mp(self, wbc_solvers_process, wbc_solver_pipes):
        self.wbc_solvers_process = wbc_solvers_process
        self.wbc_solver_pipes = wbc_solver_pipes

    def target_eef_pose_to_action_perpare(
        self,
        target_eef_pose_dict: dict,
        vel_multiplier: float = 1.0,
        env_id: int = 0,
    ):
        if self.q0s is None:
            self.q0s = (
                self.obs_buf["policy"]["joint_pos"].cpu().numpy().astype(np.float64)
            )

        assert len(target_eef_pose_dict) == 1, "UR5 only has one end-effector"

        robot: Articulation = self.scene["robot"]
        robot_world_pos = robot.data.root_pos_w - self.scene.env_origins
        robot_world_quat = robot.data.root_quat_w
        robot_world_rot = PoseUtils.matrix_from_quat(robot_world_quat)
        robot_world_pose_np = (
            PoseUtils.make_pose(robot_world_pos, robot_world_rot).cpu().numpy()
        )

        q0 = self.q0s[env_id, :6]
        target_eef_pose_dict_send = {}
        for ee, pose in target_eef_pose_dict.items():
            target_eef_pose_dict_send[ee] = pose.cpu().numpy()
        self.wbc_solver_pipes[env_id].send(
            (q0, robot_world_pose_np[env_id], target_eef_pose_dict_send, vel_multiplier)
        )
        # print("begin prepare at env {}".format(env_id))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        vel_multiplier: float = 1.0,
        env_id: int = 0,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().

            Suppose the action is [lift, head_yaw, head_pitch, eef_x_l, eef_y_l, eef_z_l, eef_quat_w_l,
            eef_quat_x_l, eef_quat_y_l, eef_quat_z_l, eef_x_r, eef_y_r, eef_z_r, eef_quat_w_r, eef_quat_x_r,
            eef_quat_y_r, eef_quat_z_r, gripper_l, gripper_r, velocity_x, velocity_y, velocity_yaw]

            pose are all relative in the robot base frame
        """
        # print("get action at env {}".format(env_id))
        success, q0 = self.wbc_solver_pipes[env_id].recv()
        self.q0s[env_id, :6] = q0[:]
        action = torch.zeros_like(self.action_manager.action)[0]
        action = torch.zeros((7,), device=action.device, dtype=action.dtype)
        action[:6] = torch.from_numpy(q0).to(action.device, action.dtype)

        action[-1] = list(gripper_action_dict.values())[0]

        # add noise to action
        if noise is not None:
            noise = noise * torch.randn_like(action)
            action[:] += noise[:]
            # action = torch.clamp(action, -1.0, 1.0)
        return action, success

    def __del__(self):
        for process in self.wbc_solvers_process:
            process.terminate()
            process.join()
        super().__del__()


class UR5PutBowlInMicroWaveAndCloseMimicJointControlMPEnv(UR5JointMimicMPEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        # signals["approch_door"] = subtask_terms["approch_door"][env_ids]
        signals["pick_bowl"] = subtask_terms["pick_bowl"][env_ids]
        signals["put_bowl_in_microwave"] = subtask_terms["put_bowl_in_microwave"][
            env_ids
        ]
        signals["close_door"] = subtask_terms["close_door"][env_ids]
        return signals


class UR5CleanPlateMimicJointControlMPEnv(UR5JointMimicMPEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["pick_fork"] = subtask_terms["pick_fork"][env_ids]
        signals["place_fork"] = subtask_terms["place_fork"][env_ids]
        signals["pick_plate"] = subtask_terms["pick_plate"][env_ids]
        signals["drop_plate"] = subtask_terms["drop_plate"][env_ids]
        signals["place_plate"] = subtask_terms["place_plate"][env_ids]
        return signals
