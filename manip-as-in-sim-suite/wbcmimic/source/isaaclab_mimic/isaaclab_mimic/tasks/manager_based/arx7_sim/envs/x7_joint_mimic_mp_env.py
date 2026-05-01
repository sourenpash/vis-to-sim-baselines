# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Sequence, Tuple

import isaaclab.sim as sim_utils
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
from isaaclab_mimic.utils.exceptions import CollisionError, WbcSolveFailedException
from isaaclab_mimic.utils.robots.wbc_controller import WbcController, WbcControllerCfg
from isaaclab_mimic.utils.robots.wbc_controller_dual import (
    DualArmWbcController,
    DualArmWbcControllerCfg,
)

from .x7_joint_mimic_env import X7BaseMimicJointWbcControlEnv


class X7BaseMimicJointWbcMpEnv(X7BaseMimicJointWbcControlEnv):
    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        ManagerBasedRLMimicEnv.__init__(self, cfg, render_mode, **kwargs)

        self.wbc_solvers = [
            DualArmWbcController(self.cfg.mimic_config.wbc_solver_cfg)
            for _ in range(self.cfg.scene.num_envs)
        ]
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[f"joint{i}" for i in range(1, 22)],
            body_names=["base_link"] + [f"link{i}" for i in range(1, 22)],
            preserve_order=True,
        )
        self.robot_entity_cfg.resolve(self.scene)
        self.q0s = None
        self.eef_name_to_link_name = {
            "l": "link10",
            "r": "link19",
        }
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.scene.stage)
        world_cfg = self.get_world_cfg()
        self.config = RobotWorldConfig.load_from_config(
            world_model=world_cfg,
            collision_activation_distance=0.5,
            collision_checker_type=CollisionCheckerType.MESH,
        )
        self.collision_world_model = RobotWorld(self.config)

    def setup_mp(self, wbc_solvers_process, wbc_solver_pipes):
        self.wbc_solvers_process = wbc_solvers_process
        self.wbc_solver_pipes = wbc_solver_pipes

    def target_eef_pose_to_action_perpare(
        self,
        target_eef_pose_dict: dict,
        vel_multiplier: float = 1.0,
        env_id: int = 0,
    ):
        robot = self.scene["robot"]
        if self.q0s is None:
            self.q0s = (
                robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].cpu().numpy()
            )

        robot: Articulation = self.scene["robot"]
        robot_world_pos = (
            robot.data.body_pos_w[:, self.robot_entity_cfg.body_ids[0], :]
            - self.scene.env_origins
        )
        robot_world_quat = robot.data.body_quat_w[
            :, self.robot_entity_cfg.body_ids[0], :
        ]
        robot_world_rot = PoseUtils.matrix_from_quat(robot_world_quat)
        robot_world_pose_np = (
            PoseUtils.make_pose(robot_world_pos, robot_world_rot).cpu().numpy()
        )

        q0 = np.concatenate(
            [np.zeros((3,), dtype=np.float32), self.q0s[env_id]], axis=0
        )
        target_eef_pose_dict_send = {}
        ee_name_in_subtask_to_urdf = {
            "l": "link10",
            "r": "link19",
        }
        for ee, pose in target_eef_pose_dict.items():
            target_eef_pose_dict_send[ee_name_in_subtask_to_urdf[ee]] = (
                pose.cpu().numpy()
            )
        self.wbc_solver_pipes[env_id].send(
            (q0, robot_world_pose_np[env_id], target_eef_pose_dict_send)
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

        success, q0 = self.wbc_solver_pipes[env_id].recv()

        self.q0s[env_id] = q0[3:]
        action = torch.zeros_like(self.action_manager.action)[0]
        action = torch.from_numpy(q0).to(action.device, action.dtype)

        action[13:15] = gripper_action_dict["l"]
        action[-2:] = gripper_action_dict["r"]

        # add noise to action
        if noise is not None:
            noise = noise * torch.randn_like(action)
            action[:15] += noise[:15]
            action[:4] += noise[:4]
            action[15:] += noise[15:]
            # action = torch.clamp(action, -1.0, 1.0)
        return action, success


class X7PickToothpasteIntoCupAndPushMimicJointWbcControlMPEnv(X7BaseMimicJointWbcMpEnv):

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["pick_toothpaste"] = subtask_terms["pick_toothpaste"][env_ids]
        signals["place_toothpaste_in_cup"] = subtask_terms["place_toothpaste_in_cup"][
            env_ids
        ]
        signals["toothpaste_in_cup_and_push"] = subtask_terms[
            "toothpaste_in_cup_and_push"
        ][env_ids]
        return signals

    def reset(self, seed=None, env_ids=None, options=None):
        super().reset(seed, env_ids, options)
        robot = self.scene["robot"]
        self.q0s = (
            robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].cpu().numpy()
        )
