# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
import omni.log
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp import ActionTerm, ActionTermCfg, JointAction, JointActionCfg
from isaaclab.utils import configclass

from .cartesian_controller import CartesianHelper

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab_mimic.utils.path import PROJECT_ROOT


class VelocityAction(ActionTerm):

    cfg: "VelocityActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "VelocityActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self.cfg = cfg
        self.env = env

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._robot_root_state = torch.zeros((self.num_envs, 7), device=self.device)
        self._robot_root_state[:, 3] = 1.0  # Initialize quaternion to identity
        self._global_step = 0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _find_nearest_angle(self, origin_angle, target_angle):
        # Find the equivalent angle within [origin_angle - pi, origin_angle + pi]
        # by removing multiples of 2π
        diff = target_angle - origin_angle
        # Normalize to [-π, π]
        normalized_diff = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi
        nearest_angle = origin_angle + normalized_diff

        return nearest_angle

    def process_actions(self, actions: torch.Tensor):
        self._processed_actions = actions
        delta_pos_r, delta_quat_r = self._compute_delta_root_transform(
            actions * self.env.physics_dt
        )
        delta_pos_w, delta_quat_w = math_utils.combine_frame_transforms(
            self._robot_root_state[:, :3],
            self._robot_root_state[:, 3:7],
            delta_pos_r,
            delta_quat_r,
        )
        self._robot_root_state[:, :2] = delta_pos_w[:, :2]
        self._robot_root_state[:, 3:7] = delta_quat_w
        # self._asset.write_root_state_to_sim(self._robot_root_state)
        target_pos = self._robot_root_state[:, :3]
        target_pos[:, 2] = self._find_nearest_angle(
            self._asset.data.joint_pos[:, 2],
            math_utils.euler_xyz_from_quat(self._robot_root_state[:, 3:7])[2],
        )
        self.target_pos = target_pos

    def _compute_delta_root_transform(
        self, actions_dt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = actions_dt[:, 0:1]
        y = actions_dt[:, 1:2]
        quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(actions_dt[:, 2]),
            torch.zeros_like(actions_dt[:, 2]),
            actions_dt[:, 2],
        )
        return torch.cat([x, y, torch.zeros_like(x)], dim=-1).to(
            torch.float32
        ), quat.to(torch.float32)

    def apply_actions(self):
        # if self._robot_root_state is None:
        #     self._robot_root_state = self._asset.data.root_state_w.clone()
        #     self._robot_root_state[:, 7:] = 0
        #     self._robot_root_state[:, :3] = self._robot_root_state[:, :3]

        self._asset.set_joint_position_target(
            self.target_pos,
            joint_ids=[0, 1, 2],
        )
        self._global_step += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        # self._robot_root_state = None
        self._robot_root_state[env_ids] = torch.zeros(
            (self.num_envs, 7), device=self.device
        )[env_ids]
        self._robot_root_state[env_ids, 3] = 1.0  # Initialize quaternion to identity


@configclass
class VelocityActionCfg(ActionTermCfg):
    """Configuration for the velocity action term."""

    class_type: type[ActionTerm] = VelocityAction

    asset_name: str = "robot"


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
            use_ik_solve_lift=cfg.use_ik_solve_lift,
        )

        self.last_joint_actions = None
        self.gripper_force = self.cfg.gripper_force

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
        self._joint_actions = []
        idxes = torch.tensor([i for i in range(21) if i not in [10, 11, 19, 20]])
        self._joint_actions = (
            self.cartesian_helper.cartesian_actions_to_joint_actions_batch(
                self.processed_actions,
                current_joints[:, 3:10],
                current_joints[:, 12:19],
                torch.arange(self.num_envs),
            )
        )
        # for i in range(len(current_joints)):
        #     self._joint_actions.append(self.cartesian_helper.cartesian_actions_to_joint_actions(self.processed_actions[i], current_joints[i][3:10], current_joints[i][12:19])[None, :])
        self._left_gripper_actions = (
            self._joint_actions[:, 10:12].sum(dim=1)[:, None] < 0.044
        )
        self._right_gripper_actions = (
            self._joint_actions[:, 19:21].sum(dim=1)[:, None] < 0.044
        )
        self._joint_actions = self._joint_actions[:, idxes]
        self._joint_actions = torch.clamp(
            self._joint_actions,
            min=self._asset.data.joint_pos_limits[:, self._joint_ids, 0][:, idxes],
            max=self._asset.data.joint_pos_limits[:, self._joint_ids, 1][:, idxes],
        )
        # if self.last_joint_actions is not None:
        #     self._joint_actions = torch.where(torch.norm(self._joint_actions - self.last_joint_actions, dim=1) > 0.1, self._joint_actions, self.last_joint_actions)
        if self.last_joint_actions is not None:
            delta_joints = self._joint_actions - self.last_joint_actions
            delta_joints = torch.clamp(
                delta_joints,
                min=-self.cfg.max_joint_velocity * self.cfg.dt,
                max=self.cfg.max_joint_velocity * self.cfg.dt,
            )
            self._joint_actions = delta_joints + self.last_joint_actions
        self.last_joint_actions = self._joint_actions

    def apply_actions(self):
        idxes = torch.tensor([i for i in range(21) if i not in [10, 11, 19, 20]])
        self._asset.set_joint_position_target(
            self._joint_actions,
            joint_ids=torch.tensor(self._joint_ids, device="cpu")[idxes].tolist(),
        )
        joint_idxes = torch.tensor([10, 11, 19, 20])
        # gripper_efforts = (
        #     1
        #     - torch.hstack(
        #         [
        #             self._left_gripper_actions,
        #             self._left_gripper_actions,
        #             self._right_gripper_actions,
        #             self._right_gripper_actions,
        #         ]
        #     )
        #     * 2
        # ) * self.gripper_force
        # self._asset.set_joint_effort_target(
        #     gripper_efforts,
        #     joint_ids=torch.tensor(self._joint_ids, device="cpu")[joint_idxes].tolist(),
        # )
        gripper_targets = (
            1
            - torch.hstack(
                [
                    self._left_gripper_actions,
                    self._left_gripper_actions,
                    self._right_gripper_actions,
                    self._right_gripper_actions,
                ]
            )
            * 1.0
        ) * 0.044
        gripper_targets = torch.clamp(gripper_targets, min=0.01, max=0.044)
        self._asset.set_joint_position_target(
            gripper_targets,
            joint_ids=torch.tensor(self._joint_ids, device="cpu")[joint_idxes].tolist(),
        )

    @property
    def action_dim(self) -> int:
        return 19


@configclass
class CartesianPositionActionCfg(JointActionCfg):
    """Configuration for the Cartesian position action term."""

    class_type: type[ActionTerm] = CartesianPositionAction

    asset_name: str = "robot"

    use_default_offset: bool = False

    urdf_path: str = f"{PROJECT_ROOT}/source/arxx7_assets/X7/urdf/X7.urdf"

    package_dirs: list[str] = [f"{PROJECT_ROOT}/source/arxx7_assets/"]

    use_in_lp_filter: bool = False

    in_lp_alpha: float = 0.9

    damp: float = 1e-6

    orientation_cost: float = 0.5

    dt: float = 1 / 120

    max_joint_velocity: float = 5

    gripper_force: float = 10.0

    use_ik_solve_lift: bool = True
    