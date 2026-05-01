# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, List

import isaaclab.utils.math as math_utils
import omni.log
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp import ActionTerm, ActionTermCfg, JointAction, JointActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab_mimic.utils.path import PROJECT_ROOT

from .cartesian_controller import CartesianHelper


class CartesianPositionAction(JointAction):

    cfg: "CartesianPositionActionCfg"

    def __init__(self, cfg: "CartesianPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()

        self.cartesian_helper = CartesianHelper(
            urdf_path=cfg.urdf_path,
            package_dirs=cfg.package_dirs,
            use_in_lp_filter=cfg.use_in_lp_filter,
            in_lp_alpha=cfg.in_lp_alpha,
            damp=cfg.damp,
            orientation_cost=cfg.orientation_cost,
            env_num=self.num_envs,
        )

        # self.last_joint_actions = None
        self.last_joint_actions = torch.zeros((self.num_envs, 7), device=self.device)
        self.has_last_joint_actions = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.gripper_force = self.cfg.gripper_force

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        current_joints = self._asset.data.joint_pos[:, self._joint_ids]
        joint_actions = []
        joint_actions = self.cartesian_helper.cartesian_actions_to_joint_actions_batch(
            self.processed_actions,
            current_joints[:, :6],
            torch.arange(self.num_envs),
        ).to(self.processed_actions.device)
        # for i in range(len(current_joints)):
        #     joint_actions.append(self.cartesian_helper.cartesian_actions_to_joint_actions(self.processed_actions[i], current_joints[i][3:10], current_joints[i][12:19])[None, :])
        gripper_actions = (joint_actions[:, 6:7] > 0.035) * 1
        joint_actions = joint_actions[:, :6]
        joint_actions = torch.cat([joint_actions, gripper_actions], dim=1)
        joint_actions = torch.clamp(
            joint_actions,
            min=self._asset.data.joint_pos_limits[:, self._joint_ids, 0][:, :7],
            max=self._asset.data.joint_pos_limits[:, self._joint_ids, 1][:, :7],
        )
        # if self.last_joint_actions is not None:
        #     joint_actions = torch.where(torch.norm(joint_actions - self.last_joint_actions, dim=1) > 0.1, joint_actions, self.last_joint_actions)
        delta_joints = joint_actions - self.last_joint_actions
        delta_joints = torch.clamp(
            delta_joints,
            min=-self.cfg.max_joint_velocity * self.cfg.dt,
            max=self.cfg.max_joint_velocity * self.cfg.dt,
        )
        joint_actions = torch.where(
            self.has_last_joint_actions.unsqueeze(1),
            delta_joints + self.last_joint_actions,
            joint_actions,
        )
        self.last_joint_actions = joint_actions
        self.has_last_joint_actions = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        applied_joint_actions = torch.cat(
            [joint_actions, torch.zeros((self.num_envs, 5)).to(joint_actions.device)],
            dim=1,
        )
        # applied_joint_actions = joint_actions
        gripper_actions = gripper_actions.reshape(-1)
        applied_joint_actions[:, 6] = gripper_actions
        applied_joint_actions[:, 7] = gripper_actions
        applied_joint_actions[:, 8] = gripper_actions * -1
        applied_joint_actions[:, 9] = gripper_actions * -1
        applied_joint_actions[:, 10] = gripper_actions * -1
        applied_joint_actions[:, 11] = gripper_actions

        self._applied_joint_actions = applied_joint_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term.

        Args:
            env_ids: The indices of the environments to reset.
        """
        if env_ids is not None:
            self.has_last_joint_actions[env_ids] = False
        else:
            self.has_last_joint_actions = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

    def apply_actions(self):
        knuckle_idx = [7, 8]
        other_idx = [i for i in range(len(self._joint_ids))]
        self._asset.set_joint_position_target(
            self._applied_joint_actions[:, other_idx],
            joint_ids=torch.tensor(
                [self._joint_ids[i] for i in other_idx], device="cpu"
            ).tolist(),
        )

        knuckle_state_to_set = self._asset.data.joint_pos[:, [6, 9]]
        self._asset.write_joint_position_to_sim(
            knuckle_state_to_set, joint_ids=knuckle_idx
        )
        # gripper_efforts = (2 * gripper_actions - 1) * self.gripper_force
        # self._asset.set_joint_effort_target(
        #     gripper_efforts,
        #     joint_ids=torch.tensor(self._joint_ids, device="cpu")[6].tolist(),
        # )

    @property
    def action_dim(self) -> int:
        return 8


# class RelativeCartesianPositionAction(CartesianPositionAction):

#     cfg: "RelativeCartesianPositionActionCfg"

#     def __init__(self, cfg: "RelativeCartesianPositionActionCfg", env: "ManagerBasedEnv"):
#         super().__init__(cfg, env)

#     def process_actions(self, actions: torch.Tensor):
#         # store the raw actions
#         self._raw_actions[:] = actions
#         # apply the affine transformations
#         self._processed_actions = self._raw_actions * self._scale + self._offset
#         # clip actions
#         if self.cfg.clip_in_cartesian_space is not None:
#             euler_actions_tuple = math_utils.euler_xyz_from_quat(self._processed_actions[:, 3:])
#             euler_actions = torch.zeros((self._env.num_envs, 3), dtype=self._processed_actions.dtype, device=self._processed_actions.device)
#             euler_actions[:, 0] = euler_actions_tuple[0]
#             euler_actions[:, 1] = euler_actions_tuple[1]
#             euler_actions[:, 2] = euler_actions_tuple[2]
#             euler_actions = torch.clamp(euler_actions, min=self.cfg.clip_in_cartesian_space[0], max=self.cfg.clip_in_cartesian_space[1])
#             self._processed_actions[:, :3] = torch.clamp(self._processed_actions[:, :3], min=self.cfg.clip_in_cartesian_space[0], max=self.cfg.clip_in_cartesian_space[1])
#             self._processed_actions[:, 3:7] = math_utils.quat_from_euler_xyz(euler_actions[:, 0], euler_actions[:, 1], euler_actions[:, 2])
#         rel_ee_pos, rel_ee_quat_curr, gripper = self._processed_actions[..., :3], self._processed_actions[..., 3:-1], self._processed_actions[..., -1:]

#         ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
#         # compute the target pose after add the relative pose
#         ee_pos_tar, ee_quat_tar = math_utils.combine_frame_transforms(
#             ee_pos_curr, ee_quat_curr, rel_ee_pos, rel_ee_quat_curr
#         )

#         current_joints = self._asset.data.joint_pos[:, self._joint_ids]
#         joint_actions = self.cartesian_helper.cartesian_actions_to_joint_actions_batch(
#             torch.cat(
#                 [ee_pos_tar, ee_quat_tar, gripper], dim=-1
#             ),
#             current_joints[:, :6],
#             torch.arange(self.num_envs),
#         ).to(self.processed_actions.device)

#         nan_idx = torch.isnan(joint_actions).any(dim=-1)
#         if torch.any(nan_idx):
#             joint_actions[nan_idx] = current_joints[nan_idx, :7] # if nan then use the current joint actions

#         # for i in range(len(current_joints)):
#         #     joint_actions.append(self.cartesian_helper.cartesian_actions_to_joint_actions(self.processed_actions[i], current_joints[i][3:10], current_joints[i][12:19])[None, :])
#         gripper_actions = (joint_actions[:, 6:7] > 0.035) * 1
#         joint_actions = joint_actions[:, :6]
#         joint_actions = torch.cat([joint_actions, gripper_actions], dim=1)
#         joint_actions = torch.clamp(
#             joint_actions,
#             min=self._asset.data.joint_pos_limits[:, self._joint_ids, 0][:, :7],
#             max=self._asset.data.joint_pos_limits[:, self._joint_ids, 1][:, :7],
#         )
#         # if self.last_joint_actions is not None:
#         #     joint_actions = torch.where(torch.norm(joint_actions - self.last_joint_actions, dim=1) > 0.1, joint_actions, self.last_joint_actions)
#         delta_joints = joint_actions - self.last_joint_actions
#         delta_joints = torch.clamp(
#             delta_joints,
#             min=-self.cfg.max_joint_velocity * self.cfg.dt,
#             max=self.cfg.max_joint_velocity * self.cfg.dt,
#         )
#         joint_actions = torch.where(
#             self.has_last_joint_actions.unsqueeze(1),
#             delta_joints + self.last_joint_actions,
#             joint_actions,
#         )
#         self.last_joint_actions = joint_actions
#         self.has_last_joint_actions = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

#         applied_joint_actions = torch.cat([joint_actions, torch.zeros((self.num_envs, 5)).to(joint_actions.device)], dim=1)
#         gripper_actions = gripper_actions.reshape(-1)
#         applied_joint_actions[:, 6] = gripper_actions
#         applied_joint_actions[:, 7] = gripper_actions
#         applied_joint_actions[:, 8] = gripper_actions * -1
#         applied_joint_actions[:, 9] = gripper_actions * -1
#         applied_joint_actions[:, 10] = gripper_actions * -1
#         applied_joint_actions[:, 11] = gripper_actions

#         self._applied_joint_actions = applied_joint_actions

#     def apply_actions(self):

#         self._asset.set_joint_position_target(
#             self._applied_joint_actions,
#             joint_ids=torch.tensor(self._joint_ids, device=self._env.device).tolist(),
#         )
#         # gripper_efforts = (2 * gripper_actions - 1) * self.gripper_force
#         # self._asset.set_joint_effort_target(
#         #     gripper_efforts,
#         #     joint_ids=torch.tensor(self._joint_ids, device="cpu")[6].tolist(),
#         # )

#     @property
#     def action_dim(self) -> int:
#         return 8

#     """
#     Helper functions.
#     """

#     def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
#         """Computes the pose of the target frame in the root frame.

#         Returns:
#             A tuple of the body's position and orientation in the root frame.
#         """
#         # obtain quantities from simulation
#         ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
#         ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
#         root_pos_w = self._asset.data.root_pos_w
#         root_quat_w = self._asset.data.root_quat_w
#         # compute the pose of the body in the root frame
#         ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
#         # account for the offset
#         if self.cfg.body_offset is not None:
#             ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
#                 ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
#             )

#         return ee_pose_b, ee_quat_b


class RelativeCartesianPositionAction(CartesianPositionAction):

    cfg: "RelativeCartesianPositionActionCfg"

    def __init__(
        self, cfg: "RelativeCartesianPositionActionCfg", env: "ManagerBasedEnv"
    ):
        self.pos_high = 0.2
        self.pos_low = -0.2
        self.rot_bound = 0.2
        super().__init__(cfg, env)

    def _clip_and_scale_actions(self, actions: torch.Tensor):
        pos_actions = actions[:, :3]
        pos_actions = torch.clamp(pos_actions, min=-1.0, max=1.0)
        pos_actions = (
            0.5 * (self.pos_high + self.pos_low)
            + 0.5 * (self.pos_high - self.pos_low) * pos_actions
        )
        rot_actions = actions[:, 3:6]
        rot_norm = torch.norm(rot_actions, dim=-1, keepdim=True)
        rot_actions = torch.where(rot_norm > 1, rot_actions / rot_norm, rot_actions)
        rot_actions = rot_actions * self.rot_bound
        gripper_actions = actions[:, -1:]
        return torch.cat([pos_actions, rot_actions, gripper_actions], dim=-1)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._clip_and_scale_actions(self._raw_actions)

        gripper = self._processed_actions[:, -1:]
        delta_pos = self._processed_actions[:, :3]
        delta_rot = self._processed_actions[:, 3:6]
        delta_quat = math_utils.quat_from_angle_axis(
            angle=torch.norm(delta_rot, dim=-1),
            axis=delta_rot / torch.norm(delta_rot, dim=-1, keepdim=True),
        )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        ee_pos_tar, ee_quat_tar = math_utils.combine_frame_transforms(
            ee_pos_curr, ee_quat_curr, delta_pos, delta_quat
        )

        current_joints = self._asset.data.joint_pos[:, self._joint_ids]
        joint_actions = self.cartesian_helper.cartesian_actions_to_joint_actions_batch(
            torch.cat([ee_pos_tar, ee_quat_tar, gripper], dim=-1),
            current_joints[:, :6],
            torch.arange(self.num_envs),
        ).to(self.processed_actions.device)

        nan_idx = torch.isnan(joint_actions).any(dim=-1)
        if torch.any(nan_idx):
            joint_actions[nan_idx] = current_joints[
                nan_idx, :7
            ]  # if nan then use the current joint actions

        # for i in range(len(current_joints)):
        #     joint_actions.append(self.cartesian_helper.cartesian_actions_to_joint_actions(self.processed_actions[i], current_joints[i][3:10], current_joints[i][12:19])[None, :])
        gripper_actions = (joint_actions[:, 6:7] > 0.0) * 1
        joint_actions = joint_actions[:, :6]
        joint_actions = torch.cat([joint_actions, gripper_actions], dim=1)
        joint_actions = torch.clamp(
            joint_actions,
            min=self._asset.data.joint_pos_limits[:, self._joint_ids, 0][:, :7],
            max=self._asset.data.joint_pos_limits[:, self._joint_ids, 1][:, :7],
        )
        # if self.last_joint_actions is not None:
        #     joint_actions = torch.where(torch.norm(joint_actions - self.last_joint_actions, dim=1) > 0.1, joint_actions, self.last_joint_actions)
        delta_joints = joint_actions - self.last_joint_actions
        delta_joints = torch.clamp(
            delta_joints,
            min=-self.cfg.max_joint_velocity * self.cfg.dt,
            max=self.cfg.max_joint_velocity * self.cfg.dt,
        )
        joint_actions = torch.where(
            self.has_last_joint_actions.unsqueeze(1),
            delta_joints + self.last_joint_actions,
            joint_actions,
        )
        self.last_joint_actions = joint_actions
        self.has_last_joint_actions = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # applied_joint_actions = torch.cat([joint_actions, torch.zeros((self.num_envs, 5)).to(joint_actions.device)], dim=1)
        applied_joint_actions = torch.cat(
            [joint_actions, torch.zeros((self.num_envs, 5)).to(joint_actions.device)],
            dim=1,
        )
        gripper_actions = gripper_actions.reshape(-1)
        applied_joint_actions[:, 6] = gripper_actions
        applied_joint_actions[:, 7] = gripper_actions
        applied_joint_actions[:, 8] = gripper_actions * -1
        applied_joint_actions[:, 9] = gripper_actions * -1
        applied_joint_actions[:, 10] = gripper_actions * -1
        applied_joint_actions[:, 11] = gripper_actions

        self._applied_joint_actions = applied_joint_actions

    def apply_actions(self):
        knuckle_idx = [7, 8]
        other_idx = [i for i in range(len(self._joint_ids))]
        self._asset.set_joint_position_target(
            self._applied_joint_actions[:, other_idx],
            joint_ids=torch.tensor(
                [self._joint_ids[i] for i in other_idx], device="cpu"
            ).tolist(),
        )

        knuckle_state_to_set = self._asset.data.joint_pos[:, [6, 9]]
        self._asset.write_joint_position_to_sim(
            knuckle_state_to_set, joint_ids=knuckle_idx
        )

        # gripper_efforts = (2 * gripper_actions - 1) * self.gripper_force
        # self._asset.set_joint_effort_target(
        #     gripper_efforts,
        #     joint_ids=torch.tensor(self._joint_ids, device="cpu")[6].tolist(),
        # )

    @property
    def action_dim(self) -> int:
        return 7

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b


@configclass
class CartesianPositionActionCfg(JointActionCfg):
    """Configuration for the Cartesian position action term."""

    class_type: type[ActionTerm] = CartesianPositionAction

    asset_name: str = "robot"

    use_default_offset: bool = False

    urdf_path: str = (
        f"{PROJECT_ROOT}/source/arxx7_assets/UR5/ur5_isaac_simulation/robot.urdf"
    )

    package_dirs: list[str] = [
        f"{PROJECT_ROOT}/source/arxx7_assets/UR5/ur5_isaac_simulation"
    ]

    use_in_lp_filter: bool = False

    in_lp_alpha: float = 0.9

    damp: float = 1e-6

    orientation_cost: float = 0.5

    dt: float = 1 / 120

    max_joint_velocity: float = 10.0

    gripper_force: float = 50.0

    body_name: str = "wrist_3_link"


@configclass
class RelativeCartesianPositionActionCfg(CartesianPositionActionCfg):
    """Configuration for the Cartesian position action term."""

    class_type: type[ActionTerm] = RelativeCartesianPositionAction

    body_offset: float = None  # TODO: check

    max_joint_velocity: float = 1.0


class GripperControl(ActionTerm):
    """Gripper Action for UR5 1.0 for open; 0.0 for close"""

    cfg: "GripperControlCfg"
    _asset: Articulation

    def __init__(self, cfg: "GripperControlCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self.cfg = cfg
        self.env = env
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, gripper_actions: torch.Tensor):
        applied_joint_actions = torch.zeros((self.num_envs, 6)).to(
            gripper_actions.device
        )
        applied_joint_actions[:, 0:1] = gripper_actions
        applied_joint_actions[:, 1:2] = gripper_actions
        applied_joint_actions[:, 2:3] = gripper_actions * -1
        applied_joint_actions[:, 3:4] = gripper_actions * -1
        applied_joint_actions[:, 4:5] = gripper_actions * -1
        applied_joint_actions[:, 5:6] = gripper_actions
        self._processed_actions = applied_joint_actions

    def apply_actions(self):
        knuckle_idx = [1, 2]
        other_idx = [i for i in range(len(self._joint_ids))]
        self._asset.set_joint_position_target(
            self._processed_actions[:, other_idx],
            joint_ids=torch.tensor(
                [self._joint_ids[i] for i in other_idx], device="cpu"
            ).tolist(),
        )
        knuckle_state_to_set = self._asset.data.joint_pos[
            :, [self._joint_ids[i] for i in [0, 3]]
        ]
        self._asset.write_joint_position_to_sim(
            knuckle_state_to_set, joint_ids=[self._joint_ids[i] for i in knuckle_idx]
        )


@configclass
class GripperControlCfg(ActionTermCfg):
    joint_names: List[str] = MISSING  # only for robotiq, just name can change
    class_type = GripperControl
