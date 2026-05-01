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

DEBUG = False


class UR5BaseMimicJointControlEnv(ManagerBasedRLMimicEnv):
    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.wbc_solvers = [
            WbcController(self.cfg.mimic_config.wbc_solver_cfg)
            for _ in range(self.cfg.scene.num_envs)
        ]
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

        if DEBUG:
            from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.goal_l_marker = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/goal_l_marker")
            )

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
        if self.q0s is None:
            self.q0s = (
                self.obs_buf["policy"]["joint_pos"].cpu().numpy().astype(np.float64)
            )

        assert len(target_eef_pose_dict) == 1, "UR5 only has one end-effector"

        action = torch.zeros_like(self.action_manager.action)[0]
        robot: Articulation = self.scene["robot"]
        robot_world_pos = robot.data.root_pos_w - self.scene.env_origins
        robot_world_quat = robot.data.root_quat_w
        robot_world_rot = PoseUtils.matrix_from_quat(robot_world_quat)
        robot_world_pose_np = (
            PoseUtils.make_pose(robot_world_pos, robot_world_rot).cpu().numpy()
        )

        q0 = self.q0s[env_id, :6]
        self.wbc_solvers[env_id].update_joint_pos(q0)
        self.wbc_solvers[env_id].update_root_pose(robot_world_pose_np[env_id])

        for ee, target_eef_pose in target_eef_pose_dict.items():
            self.wbc_solvers[env_id].set_goal(target_eef_pose.cpu().numpy())
            if DEBUG:
                self.goal_l_marker.visualize(
                    target_eef_pose[None, :3, 3],
                    PoseUtils.quat_from_matrix(target_eef_pose[None, :3, :3]),
                )

        try:
            success, qd = self.wbc_solvers[env_id].step_robot(
                vel_multiplier=vel_multiplier,
                # distance=d,
                # norm_a2b=d_vec,
                # index_a=np.arange(0, 22),
                # body_names=self.robot_entity_cfg.body_names
            )
        except WbcSolveFailedException:
            print("WbcSolveFailedException at env {}".format(env_id))
            success = True
            qd = np.zeros_like(q0)
        q0 = q0 + qd * self.step_dt
        self.q0s[env_id, :6] = q0[:]
        action = torch.zeros((7,), device=action.device, dtype=action.dtype)
        action[:6] = torch.from_numpy(q0).to(action.device, action.dtype)

        action[-1] = list(gripper_action_dict.values())[0]

        # add noise to action
        if noise is not None:
            noise = noise * torch.randn_like(action)
            action[:] += noise[:]
            # action = torch.clamp(action, -1.0, 1.0)
        return action, success

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

        robot = self.scene["robot"]

        robot_root_pos = robot.data.root_pos_w - self.scene.env_origins
        robot_root_quat = robot.data.root_quat_w

        target_pos = action[:, :3]
        target_quat = action[:, 3:7]
        target_pos_in_world, target_quat_in_world = PoseUtils.combine_frame_transforms(
            robot_root_pos, robot_root_quat, target_pos, target_quat
        )
        target_rot_in_world = PoseUtils.matrix_from_quat(target_quat_in_world)
        target_poses_in_world = PoseUtils.make_pose(
            target_pos_in_world, target_rot_in_world
        ).clone()

        return {eef_name: target_poses_in_world}

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
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3],
                PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
            )
        for obj_name, obj_state in articulation_states.items():
            if obj_name == "robot":
                continue
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3],
                PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
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


class UR5CloseMicroWaveMimicJointControlEnv(UR5BaseMimicJointControlEnv):
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


class UR5PutBowlInMicroWaveAndCloseMimicJointControlEnv(UR5BaseMimicJointControlEnv):
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


class UR5CleanPlateMimicJointControlEnv(UR5BaseMimicJointControlEnv):
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
