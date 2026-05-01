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

DEBUG = False


class X7BaseMimicJointWbcControlEnv(ManagerBasedRLMimicEnv):
    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

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

        if DEBUG:
            from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.goal_l_marker = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/goal_l_marker")
            )
            self.goal_r_marker = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path="/Visuals/goal_r_marker")
            )

    def get_world_cfg(self):
        return self.usd_helper.get_obstacles_from_stage(
            reference_prim_path="/World",
            only_paths=sim_utils.find_matching_prim_paths("/World/envs/env.*/scene"),
        ).get_collision_check_world()

    def update_obstacles(self):
        self.collision_world_model.update_world(self.get_world_cfg())

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

        robot: Articulation = self.scene["robot"]
        robot_root_pos = (
            robot.data.body_pos_w[:, self.robot_entity_cfg.body_ids[0], :]
            - self.scene.env_origins
        )
        robot_root_quat = robot.data.body_quat_w[
            :, self.robot_entity_cfg.body_ids[0], :
        ]

        eef_pos_in_world, eef_quat_in_world = PoseUtils.combine_frame_transforms(
            robot_root_pos[env_ids], robot_root_quat[env_ids], eef_pos, eef_quat
        )
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(
            eef_pos_in_world, PoseUtils.matrix_from_quat(eef_quat_in_world)
        )

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

        self.cfg: "X7PourWaterMimicEnvCfg"
        if self.q0s is None:
            self.q0s = (
                self.obs_buf["policy"]["joint_pos"].cpu().numpy().astype(np.float64)
            )

        action = torch.zeros_like(self.action_manager.action)[0]
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
        self.wbc_solvers[env_id].update_joint_pos(q0)
        self.wbc_solvers[env_id].update_root_pose(robot_world_pose_np[env_id])

        for ee, target_eef_pose in target_eef_pose_dict.items():
            self.wbc_solvers[env_id].set_goal(
                self.eef_name_to_link_name[ee], target_eef_pose.cpu().numpy()
            )
            if DEBUG:
                if ee == "l":
                    self.goal_l_marker.visualize(
                        target_eef_pose[None, :3, 3],
                        PoseUtils.quat_from_matrix(target_eef_pose[None, :3, :3]),
                    )
                else:
                    self.goal_r_marker.visualize(
                        target_eef_pose[None, :3, 3],
                        PoseUtils.quat_from_matrix(target_eef_pose[None, :3, :3]),
                    )

        xb = robot.data.body_link_pos_w.clone()[
            env_id : env_id + 1, self.robot_entity_cfg.body_ids, :
        ]
        xb = torch.cat(
            [xb[:, :, None, :], torch.ones_like(xb[:, :, None, 0:1])], dim=-1
        )
        d, d_vec = self.collision_world_model.get_collision_vector(xb.to("cuda"))
        d = d.view(-1).cpu()
        mask = d != 0.0
        # if not mask.all().item():
        #     raise CollisionError("Collision detected")

        if len(d) > 0 and d_vec is not None:
            # if DEBUG:
            #     print(d[d.argmin()])
            #     draw_line_min(list(robot.data.body_link_pos_w.clone()[env_id, self.robot_entity_cfg.body_ids, :].cpu().numpy()),
            #                 list(d_vec[..., :3].view(-1, 3).cpu().numpy()), d.argmin())
            d_vec = d_vec[..., :3]
            d_vec = d_vec[:, mask, ...]
            d_vec = d_vec.view(-1, 3).cpu()
        else:
            d = None
            d_vec = None
        try:
            success, qd = self.wbc_solvers[env_id].step_robot(
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
        q0[:3] = qd[:3]
        self.q0s[env_id] = q0[3:]
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

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Converts action during recording (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action when action is same during recording and generating (in wbc maybe in not like this). Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """

        eef_names = list(self.cfg.subtask_configs.keys())

        robot = self.scene["robot"]

        robot_root_pos = (
            robot.data.body_pos_w[:, self.robot_entity_cfg.body_ids[0], :]
            - self.scene.env_origins
        )
        robot_root_quat = robot.data.body_quat_w[
            :, self.robot_entity_cfg.body_ids[0], :
        ]

        target_eef_pose_dict = dict()
        if "l" in eef_names:
            target_pos = action[:, 3:6]
            target_quat = action[:, 6:10]
            target_pos_in_world, target_quat_in_world = (
                PoseUtils.combine_frame_transforms(
                    robot_root_pos, robot_root_quat, target_pos, target_quat
                )
            )
            target_rot_in_world = PoseUtils.matrix_from_quat(target_quat_in_world)
            target_poses_in_world = PoseUtils.make_pose(
                target_pos_in_world, target_rot_in_world
            ).clone()
            target_eef_pose_dict["l"] = target_poses_in_world
        if "r" in eef_names:
            target_pos = action[:, 10:13]
            target_quat = action[:, 13:17]
            target_pos_in_world, target_quat_in_world = (
                PoseUtils.combine_frame_transforms(
                    robot_root_pos, robot_root_quat, target_pos, target_quat
                )
            )
            target_rot_in_world = PoseUtils.matrix_from_quat(target_quat_in_world)
            target_poses_in_world = PoseUtils.make_pose(
                target_pos_in_world, target_rot_in_world
            ).clone()
            target_eef_pose_dict["r"] = target_poses_in_world

        return target_eef_pose_dict

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
        eef_names = list(self.cfg.subtask_configs.keys())

        gripper_actions_dict = dict()
        if "l" in eef_names:
            gripper_actions_dict["l"] = actions[:, -5]
        if "r" in eef_names:
            gripper_actions_dict["r"] = actions[:, -4]
        return gripper_actions_dict

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

        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3],
                PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
            )
        return object_pose_matrix


class X7PourWaterMimicJointWbcControlEnv(X7BaseMimicJointWbcControlEnv):
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


class X7PickWineMimicJointWbcControlEnv(X7BaseMimicJointWbcControlEnv):
    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["pick_wine"] = subtask_terms["pick_wine"][env_ids]
        return signals


class X7ApproachWineMimicJointWbcControlEnv(X7BaseMimicJointWbcControlEnv):

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["approach_wine"] = subtask_terms["approach_wine"][env_ids]
        return signals


class X7PickToothpasteIntoCupAndPushMimicJointWbcControlEnv(
    X7BaseMimicJointWbcControlEnv
):

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
        res = super().reset(seed, env_ids, options)
        self.q0s = None
        return res
