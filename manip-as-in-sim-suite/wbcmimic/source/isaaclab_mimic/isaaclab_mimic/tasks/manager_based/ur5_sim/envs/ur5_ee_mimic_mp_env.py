# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_mimic.envs import ManagerBasedRLMimicEnv


class UR5BaseMimicEEControlEnv(ManagerBasedRLMimicEnv):
    """
    UR5 Pour Water Mimic Environment
    """

    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg: "UR5PourWaterMimicEnvCfg"
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
        eef_pos = self.obs_buf["policy"][f"eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"][f"eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        env_id: int = 0,
        vel_multiplier: float = 1.0,
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

        self.cfg: "UR5PourWaterMimicEnvCfg"
        actions = torch.zeros_like(self.action_manager.action)

        robot = self.scene["robot"]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        actions[:, :7] = torch.cat(
            [target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0
        )
        actions[:, -1] = gripper_action

        # add noise to action
        if noise is not None:
            noise = noise * torch.randn_like(actions)
            actions += noise
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

        target_pos = action[:, :3]
        target_quat = action[:, 3:7]

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

        return {eef_name: actions[:, -1]}

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
        articulation_states = asset_states["articulation"]
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
        for obj_name, obj_state in articulation_states.items():
            if obj_name == "robot":
                continue
            object_pose_matrix[obj_name] = PoseUtils.pose_in_A_to_pose_in_B(
                pose_in_A=PoseUtils.make_pose(
                    obj_state["root_pose"][env_ids, :3],
                    PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
                ),
                pose_A_in_B=PoseUtils.pose_inv(robot_pose_matrix),
            )
        return object_pose_matrix

    @abstractmethod
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
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        return signals


class UR5CloseMicroWaveMimicEEControlEnv(UR5BaseMimicEEControlEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        # signals["approch_door"] = subtask_terms["approch_door"][env_ids]
        signals["close_door"] = subtask_terms["close_door"][env_ids]
        return signals


class UR5PutBowlInMicroWaveAndCloseMimicEEControlEnv(UR5BaseMimicEEControlEnv):
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


class UR5CleanPlateEEControlEnv(UR5BaseMimicEEControlEnv):
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
