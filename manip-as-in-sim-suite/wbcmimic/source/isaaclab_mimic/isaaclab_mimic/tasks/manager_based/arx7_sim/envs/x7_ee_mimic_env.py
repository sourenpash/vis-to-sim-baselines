# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Sequence, Tuple

import isaaclab.utils.math as PoseUtils
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_mimic.envs import ManagerBasedRLMimicEnv

DEBUG = False


class X7BaseMimicEEControlEnv(ManagerBasedRLMimicEnv):
    """
    X7 Pour Water Mimic Environment
    """

    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.left = self.cfg.mimic_config.left
        if self.left:
            self.robot_entity_cfg = SceneEntityCfg(
                "robot",
                joint_names=[f"joint{i}" for i in range(4, 11)],
                preserve_order=True,
            )
        else:
            self.robot_entity_cfg = SceneEntityCfg(
                "robot",
                joint_names=[f"joint{i}" for i in range(13, 20)],
                preserve_order=True,
            )

        self.robot_entity_cfg.resolve(self.scene)

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector. (l or r)
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"][f"eef_pos_{eef_name}"][env_ids]
        eef_quat = self.obs_buf["policy"][f"eef_quat_{eef_name}"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        vel_multiplier: float = 1.0,
        env_id: int = 0,
    ) -> torch.Tensor:
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

        self.cfg: "X7PourWaterMimicEnvCfg"
        actions = torch.zeros_like(self.action_manager.action)
        actions[:, 0] = self.cfg.mimic_config.lift
        actions[:, 1] = self.cfg.mimic_config.head_yaw
        actions[:, 2] = self.cfg.mimic_config.head_pitch

        robot = self.scene["robot"]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        if self.left:
            actions[:, 3:10] = torch.cat(
                [target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0
            )
            actions[:, 10:17] = torch.tensor(
                [0.1787, -0.5602, -0.1007, 1, 0, 0, 0], device=actions.device
            )
            actions[:, -5] = gripper_action
            actions[:, -4] = 0.044
        else:
            actions[:, 10:17] = torch.cat(
                [target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0
            )
            actions[:, 3:10] = torch.tensor(
                [0.1787, 0.5602, -0.1007, 1, 0, 0, 0], device=actions.device
            )
            actions[:, -4] = gripper_action
            actions[:, -5] = 0.044

        # add noise to action
        if noise is not None:
            noise = noise * torch.randn_like(actions)
            if self.left:
                actions[:, 3:10] += noise[:, 3:10]
                actions[:, -5] += noise[:, -5]
            else:
                actions[:, 10:17] += noise[:, 10:17]
                actions[:, -4] += noise[:, -4]
            actions = torch.clamp(actions, -1.0, 1.0)
        return actions

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """

        eef_name = list(self.cfg.subtask_configs.keys())[0]

        if self.left:
            target_pos = action[:, 3:6]
            target_quat = action[:, 6:10]
        else:
            target_pos = action[:, 10:13]
            target_quat = action[:, 13:17]

        target_rot = PoseUtils.matrix_from_quat(target_quat)
        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        # last dimension is gripper action
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        if self.left:
            return {eef_name: actions[:, -5]}
        else:
            return {eef_name: actions[:, -4]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        raise NotImplementedError

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        asset_states = self.scene.get_state(is_relative=True)

        rigid_object_states = asset_states["rigid_object"]
        robot_states = asset_states["articulation"]["robot"]
        robot_pose_matrix = PoseUtils.make_pose(
            robot_states["root_pose"][env_ids, :3],
            PoseUtils.matrix_from_quat(robot_states["root_pose"][env_ids, 3:7]),
        )
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_matrix[obj_name] = PoseUtils.pose_in_A_to_pose_in_B(
                pose_in_A=PoseUtils.make_pose(
                    obj_state["root_pose"][env_ids, :3],
                    PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
                ),
                pose_A_in_B=PoseUtils.pose_inv(robot_pose_matrix),
            )
        return object_pose_matrix


class X7PourWaterMimicEEControlEnv(X7BaseMimicEEControlEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["pick_pour_item"] = subtask_terms["pick_pour_item"][env_ids]
        signals["pour"] = subtask_terms["pour"][env_ids]
        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
        return signals


class X7ApproachWineMimicEEControlEnv(X7BaseMimicEEControlEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["approach_wine"] = subtask_terms["approach_wine"][env_ids]
        return signals


class X7PickWineMimicEEControlEnv(X7BaseMimicEEControlEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["pick_wine"] = subtask_terms["pick_wine"][env_ids]
        return signals


class X7LineObjectsMimicEEControlEnv(X7BaseMimicEEControlEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["pick_item_1"] = subtask_terms["pick_item_1"][env_ids]
        signals["place_item_1"] = subtask_terms["place_item_1"][env_ids]
        signals["pick_item_2"] = subtask_terms["pick_item_2"][env_ids]
        signals["place_item_2"] = subtask_terms["place_item_2"][env_ids]
        signals["pick_item_3"] = subtask_terms["pick_item_3"][env_ids]
        signals["place_item_3"] = subtask_terms["place_item_3"][env_ids]
        signals["pick_item_4"] = subtask_terms["pick_item_4"][env_ids]
        signals["place_item_4"] = subtask_terms["place_item_4"][env_ids]
        return signals


class X7PickToothpasteIntoCupAndPushMimicEEControlEnv(X7BaseMimicEEControlEnv):

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
