# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import einops
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, ArticulationData, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import (
    ManagerTermBase,
    ManagerTermBaseCfg,
    RewardTermCfg,
    SceneEntityCfg,
)
from isaaclab.sensors import Camera, FrameTransformer, FrameTransformerData, TiledCamera
from isaaclab_mimic.utils.pcd_utils import (
    create_pointcloud_from_rgbd_batch,
    pcd_downsample_torch,
)
from isaaclab_mimic.utils.prim import get_points_at_path
from isaacsim.core.prims import XFormPrim
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from isaaclab_mimic.envs import WbcSubTaskConfig


class pcd_observation:
    def __init__(self):
        self.initialized: bool = False
        self.pointnet = None
        # TODO initialize pointnet

    def _initialize(self, env, objs: List[SceneEntityCfg]):
        self.initialized = True
        self.obj_points = {}

        for obj in objs:
            obj: RigidObject | Articulation = env.scene[a.name]
            obj_prim_path = str(XFormPrim(obj_a.cfg.prim_path).prims[0].GetPrimPath())
            points, faces = get_points_at_path(obj_prim_path, return_faces=True)
            pcds = generate_point_cloud(points, faces)
            torch.from_numpy(pcds).to(env.device).repeat((env.num_envs, 1, 1))

    def __call__(
        self, env: ManagerBasedRLEnv, objs: List[SceneEntityCfg], n_points: int
    ):
        if not self.initialized:
            self._initialize(env, objs)

        pcd_obs = []
        for obj in objs:
            obj: RigidObject | Articulation = env.scene[a.name]
            obj.resolve(env.scene)
            if obj.body_names:
                obj_id = obj.body_ids[0]
                obj_pose = obj.data.body_state_w[:, a_id, :7]
            else:
                if len(obj.data.root_pos_w.shape) == 2:
                    obj_pose = obj.data.root_pos_w
                else:
                    obj_pose = obj.data.root_pos_w[..., 0, :]
            # TODO: transform the obj pcd with the pose data
            obj_pcd = transform(self.obj_points[obj], obj_pose)  # (env, N, 3)
            assert obj_pcd.shape == (env.num_envs, 3)
            pcd_obs.append(obj_pcd)

        pcd_obs = torch.concatenate(pcd_obs, axis=1)
        # TODO: uniform sample
        pcd_obs = uniform_sample(pcd_obs, n_points)

        point_fea = self.pointnet(point_fea)

        return point_fea


class reset_joints_close_to_object(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cartesian_helper = None
        self.__name__ = "reset_joints_close_to_object"

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: List[int],
        action_cfg,
        object_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=["wrist_3_link"]
        ),
        threshold: float = 0.5,
        min_height: float = 0.1,
    ):
        """Reset the robot joints to a position close to the object."""
        if self.cartesian_helper is None:
            from .cartesian_controller import CartesianHelper

            self.cartesian_helper = CartesianHelper(
                urdf_path=action_cfg.cartesian.urdf_path,
                package_dirs=action_cfg.cartesian.package_dirs,
                use_in_lp_filter=action_cfg.cartesian.use_in_lp_filter,
                in_lp_alpha=action_cfg.cartesian.in_lp_alpha,
                damp=action_cfg.cartesian.damp,
                orientation_cost=action_cfg.cartesian.orientation_cost,
                env_num=len(env_ids),
            )
        robot: Articulation = env.scene[robot_cfg.name]
        robot_cfg.resolve(env.scene)
        ee_id = robot_cfg.body_ids[0]

        object: RigidObject = env.scene[object_cfg.name]
        robot_cfg.resolve(env.scene)
        object_cfg.resolve(env.scene)

        robot_joints = robot.data.joint_pos[env_ids, robot_cfg.joint_ids]
        robot_pos = robot.data.root_pos_w[env_ids, :]
        robot_quat = robot.data.root_state_w[
            env_ids, 3:7
        ]  # robot root orientation in world frame
        object_pos = object.data.root_pos_w[env_ids, :]
        ee_pose_w = robot.data.body_state_w[env_ids, ee_id, 0:7]
        ee_quat_w = ee_pose_w[:, 3:7]

        def quat_from_two_vectors(v1, v2):
            """计算将单位向量 v1 旋转到单位向量 v2 的四元数"""
            dot = (v1 * v2).sum(dim=-1, keepdim=True)  # (N, 1)

            # 条件分支掩码
            same = dot > 0.999
            opposite = dot < -0.999

            # 默认情况（一般旋转）
            axis = torch.cross(v1, v2, dim=-1)
            axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
            angle = torch.acos(dot.clamp(-1.0, 1.0))
            sin_half = torch.sin(angle / 2)
            cos_half = torch.cos(angle / 2)
            q_general = torch.cat([cos_half, sin_half * axis], dim=-1)

            # 对于反方向的情形，选一个正交轴
            ortho = torch.zeros_like(v1)
            ortho[:, 0] = 1.0
            mask = torch.allclose(v1, ortho, atol=1e-3)
            if mask:
                ortho[:, 0] = 0.0
                ortho[:, 1] = 1.0
            axis_ortho = torch.cross(v1, ortho, dim=-1)
            axis_ortho = axis_ortho / torch.linalg.norm(
                axis_ortho, dim=-1, keepdim=True
            )
            q_opposite = torch.cat(
                [torch.zeros((v1.shape[0], 1), device=v1.device), axis_ortho], dim=-1
            )

            q_result = torch.where(
                same,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=v1.device).expand(
                    v1.shape[0], 4
                ),
                q_general,
            )
            q_result = torch.where(opposite, q_opposite, q_result)
            q_result = q_result / torch.linalg.norm(q_result, dim=-1, keepdim=True)
            return math_utils.quat_unique(q_result)

        # Compute target pose in world frame
        # 1. Generate random position
        random_delta_ee_pos = torch.randn_like(object_pos)
        random_delta_ee_pos = random_delta_ee_pos / torch.linalg.norm(
            random_delta_ee_pos, dim=-1, keepdim=True
        )
        random_delta_ee_pos[:, 2] = torch.abs(random_delta_ee_pos[:, 2])
        tar_ee_pos_w = object_pos + random_delta_ee_pos * threshold
        tar_ee_pos_w[:, 2] = tar_ee_pos_w[:, 2] + min_height

        # 2. Compute target orientation in world frame
        direction_w = object_pos - tar_ee_pos_w
        direction_w = direction_w / torch.linalg.norm(direction_w, dim=-1, keepdim=True)
        z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=robot_joints.device).repeat(
            len(env_ids), 1
        )
        current_z = math_utils.quat_rotate(ee_quat_w, z_axis_local)
        delta_quat_w = quat_from_two_vectors(current_z, direction_w)
        tar_ee_quat_w = math_utils.quat_mul(delta_quat_w, ee_quat_w)
        tar_ee_quat_w = tar_ee_quat_w / torch.linalg.norm(
            tar_ee_quat_w, dim=-1, keepdim=True
        )
        tar_ee_quat_w = math_utils.quat_unique(tar_ee_quat_w)

        # Transform from world frame to robot frame
        # 1. Transform position
        tar_ee_pos_r = tar_ee_pos_w - robot_pos  # translate
        tar_ee_pos_r = math_utils.quat_rotate(
            math_utils.quat_conjugate(robot_quat), tar_ee_pos_r
        )  # rotate
        # 2. Transform orientation
        tar_ee_quat_r = math_utils.quat_mul(
            math_utils.quat_conjugate(robot_quat), tar_ee_quat_w
        )

        current_joints = robot.data.joint_pos[env_ids, robot_cfg.joint_ids]

        joint_pos = torch.zeros(
            (len(env_ids), 7), dtype=current_joints.dtype, device=current_joints.device
        )
        for i in range(len(env_ids)):
            joint_pos[i, :] = (
                self.cartesian_helper.cartesian_actions_to_joint_actions_batch(
                    torch.cat(
                        [
                            tar_ee_pos_r,
                            tar_ee_quat_r,
                            torch.zeros(*tar_ee_quat_r.shape, device=object_pos.device),
                        ],
                        dim=-1,
                    )[i : i + 1, :],
                    current_joints[i : i + 1, :6],
                    torch.arange(1),
                )
            )
        nan_idx = torch.isnan(joint_pos).any(dim=-1)
        if torch.any(nan_idx):
            joint_pos[nan_idx, :6] = current_joints[nan_idx, :6]
        joint_pos = torch.cat(
            [joint_pos, torch.zeros(joint_pos.shape[0], 5, device=joint_pos.device)],
            dim=-1,
        )  # the robotiq gripper has 6 joints
        joint_vel = torch.zeros_like(joint_pos)

        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def rel_a_to_b_distance(
    env: ManagerBasedRLEnv, a: SceneEntityCfg, b: SceneEntityCfg
) -> torch.Tensor:
    """The distance between the a and b."""
    rel_pos = rel_a_to_b_position(env, a, b)

    return torch.linalg.norm(rel_pos, dim=-1)


def rel_a_to_b_position(
    env: ManagerBasedRLEnv, a: SceneEntityCfg, b: SceneEntityCfg
) -> torch.Tensor:
    """The position of a relative to b."""

    def get_pos_w(obj, entity_cfg: SceneEntityCfg) -> torch.Tensor:
        if isinstance(obj, Articulation):
            entity_cfg.resolve(env.scene)
            return obj.data.body_state_w[:, entity_cfg.body_ids, 0:3].mean(dim=-2)
        elif isinstance(obj, RigidObject):
            return obj.data.root_pos_w
        elif isinstance(obj, FrameTransformer):
            return obj.data.target_pos_w[:, 0, :]
        else:
            raise NotImplementedError

    obj_a = env.scene[a.name if isinstance(a, SceneEntityCfg) else a]
    obj_b = env.scene[b.name if isinstance(b, SceneEntityCfg) else b]
    obj_a_pos = get_pos_w(obj_a, a)
    obj_b_pos = get_pos_w(obj_b, b)
    return obj_a_pos - obj_b_pos


def rel_a_to_b_orn(
    env: ManagerBasedRLEnv, a: SceneEntityCfg, b: SceneEntityCfg, type="quat"
) -> torch.Tensor:
    """The orientation of a relative to b.

    The quaternion is made unique by ensuring the real part is positive.
    """

    def get_pose_w(obj, entity_cfg: SceneEntityCfg) -> torch.Tensor:
        if isinstance(obj, Articulation):
            entity_cfg.resolve(env.scene)
            return obj.data.body_state_w[:, entity_cfg.body_ids, 0:7].mean(dim=-2)
        elif isinstance(obj, RigidObject):
            return torch.cat([obj.data.root_pos_w, obj.data.root_quat_w], dim=-1)
        elif isinstance(obj, FrameTransformer):
            return torch.cat(
                [obj.data.target_pos_w[:, 0, :], obj.data.target_quat_w[:, 0, :]],
                dim=-1,
            )
        else:
            raise NotImplementedError

    obj_a = env.scene[a.name if isinstance(a, SceneEntityCfg) else a]
    obj_b = env.scene[b.name if isinstance(b, SceneEntityCfg) else b]
    obj_a_pose = get_pose_w(obj_a, a)
    obj_b_pose = get_pose_w(obj_b, b)

    if type == "euler":
        raise NotImplementedError
    else:
        res = torch.cat(
            math_utils.subtract_frame_transforms(
                obj_b_pose[:, 0:3],
                obj_b_pose[:, 3:7],
                obj_a_pose[:, 0:3],
                obj_a_pose[:, 3:7],
            ),
            dim=1,
        )
        return math_utils.quat_unique(res[:, 3:])


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[
        ..., 1:, :
    ] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def ee_euler(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    The euler angles are in the order of roll, pitch, yaw.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]


class lift_success_simple(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: ManagerTermBaseCfg):
        # initialize the base class
        super().__init__(cfg, env)
        self.origin_height: torch.Tensor = (
            torch.ones(env.num_envs, device=env.device) * 0.42
        )  # default origin height

    def reset(self, env_ids: torch.Tensor = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        item_cfg: SceneEntityCfg = self.cfg.params["item_cfg"]
        item_pos = self._env.scene[item_cfg.name].data.root_pos_w
        self.origin_height[env_ids] = item_pos[env_ids, 2].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        item_cfg: SceneEntityCfg,
        height_threshold: float = 0.1,
    ):
        item_pos = env.scene[item_cfg.name].data.root_pos_w
        self.origin_height = torch.where(
            self.origin_height < item_pos[:, 2], self.origin_height, item_pos[:, 2]
        )
        return item_pos[:, 2] > height_threshold + self.origin_height


def item_out_of_bounds(
    env: ManagerBasedRLEnv,
    item_cfg: SceneEntityCfg,
    bounds: dict[str, tuple[float, float]],
):
    item_pos = env.scene[item_cfg.name].data.root_pos_w
    item_pos_local = item_pos - env.scene.env_origins
    out_of_bounds = torch.zeros(env.num_envs, device=env.device)
    for key in bounds:
        if key == "x":
            out_of_bounds = torch.logical_or(
                out_of_bounds,
                torch.logical_or(
                    item_pos_local[:, 0] < bounds["x"][0],
                    item_pos_local[:, 0] > bounds["x"][1],
                ),
            )
        elif key == "y":
            out_of_bounds = torch.logical_or(
                out_of_bounds,
                torch.logical_or(
                    item_pos_local[:, 1] < bounds["y"][0],
                    item_pos_local[:, 1] > bounds["y"][1],
                ),
            )
        elif key == "z":
            out_of_bounds = torch.logical_or(
                out_of_bounds,
                torch.logical_or(
                    item_pos_local[:, 2] < bounds["z"][0],
                    item_pos_local[:, 2] > bounds["z"][1],
                ),
            )
    return out_of_bounds


class lift_success(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: ManagerTermBaseCfg):
        super().__init__(cfg, env)
        self.init_item_pos = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        item_pos = self._env.scene[self.cfg.params["item_cfg"].name].data.root_pos_w
        self.init_item_pos[env_ids] = item_pos[env_ids, :3]

    def __call__(
        self,
        env,
        item_cfg: SceneEntityCfg,
        robot_grasp_cfg: SceneEntityCfg,
        height_threshold: float = 0.1,
        horizontal_threshold: float = 0.05,
        max_height: float = 0.1,
    ):
        is_grasped = is_object_grasped(env, robot_grasp_cfg)
        item_pos = env.scene[item_cfg.name].data.root_pos_w
        item_horizontal_movement = torch.norm(
            item_pos[:, :2] - self.init_item_pos[:, :2], dim=-1
        )
        delta_heights = item_pos[:, 2:3].squeeze(-1) - self.init_item_pos[
            :, 2:3
        ].squeeze(-1)
        rew = delta_heights > height_threshold
        rew[~is_grasped] = 0.0
        rew[item_horizontal_movement > horizontal_threshold] = 0.0
        rew[delta_heights > max_height] = 0.0
        return rew


# def lift_success(
#     env: ManagerBasedRLEnv,
#     item_cfg: SceneEntityCfg,
#     robot_cfg: SceneEntityCfg,
#     threshold: float = 0.01,
#     origin_height: float = 0.42,
#     height_threshold: float = 0.1,
# ):
#     item_pos = env.scene[item_cfg.name].data.root_pos_w
#     robot = env.scene[robot_cfg.name]
#     robot_cfg.resolve(env.scene)
#     robot_ee_pos = robot.data.body_state_w[:, robot_cfg.body_ids, 0:3].mean(dim=-2)
#     dist = torch.norm(item_pos[:, :2] - robot_ee_pos[:, :2], dim=-1)[:, None]
#     return torch.logical_and(
#         dist < threshold, item_pos[:, 2:3] > origin_height + height_threshold
#     ).squeeze(-1)


def approach_success_frame_transformer(
    env: ManagerBasedRLEnv,
    frame_transformer_name: str,
    threshold: float = 0.01,
    to_print: bool = False,
):
    frame_helper = env.scene[frame_transformer_name]
    dist = torch.norm(
        frame_helper.data.target_pos_w[:, 0, :]
        - frame_helper.data.target_pos_w[:, 1, :],
        dim=-1,
    )
    if to_print:
        print(
            f"Dist between {frame_helper.data.target_frame_names[0]} and {frame_helper.data.target_frame_names[1]}: {dist}"
        )
    return dist < threshold


def stay_away(
    env: ManagerBasedRLEnv,
    obj_1_cfg: SceneEntityCfg | str,
    obj_2_cfg: SceneEntityCfg | str,
    threshold: float = 0.1,
    to_print: bool = False,
):
    def get_pos_w(obj, entity_cfg: SceneEntityCfg) -> torch.Tensor:
        if isinstance(obj, Articulation):
            entity_cfg.resolve(env.scene)
            return obj.data.body_state_w[:, entity_cfg.body_ids, 0:3].mean(dim=-2)
        elif isinstance(obj, RigidObject):
            return obj.data.root_pos_w
        elif isinstance(obj, FrameTransformer):
            return obj.data.target_pos_w[:, 0, :]
        else:
            raise NotImplementedError

    obj_1 = env.scene[
        obj_1_cfg.name if isinstance(obj_1_cfg, SceneEntityCfg) else obj_1_cfg
    ]
    obj_2 = env.scene[
        obj_2_cfg.name if isinstance(obj_2_cfg, SceneEntityCfg) else obj_2_cfg
    ]
    obj_1_pos = get_pos_w(obj_1, obj_1_cfg)
    obj_2_pos = get_pos_w(obj_2, obj_2_cfg)
    dist = torch.norm(obj_1_pos - obj_2_pos, dim=-1)
    if to_print:
        print(f"Dist between {obj_1_cfg.name} and {obj_2_cfg.name}: {dist}")
    return dist > threshold


def approach_success(
    env: ManagerBasedRLEnv,
    obj_1_cfg: SceneEntityCfg | str,
    obj_2_cfg: SceneEntityCfg | str,
    threshold: float = 0.01,
    to_print: bool = False,
):
    def get_pos_w(obj, entity_cfg: SceneEntityCfg) -> torch.Tensor:
        if isinstance(obj, Articulation):
            entity_cfg.resolve(env.scene)
            return obj.data.body_state_w[:, entity_cfg.body_ids, 0:3].mean(dim=-2)
        elif isinstance(obj, RigidObject):
            return obj.data.root_pos_w
        elif isinstance(obj, FrameTransformer):
            return obj.data.target_pos_w[:, 0, :]
        else:
            raise NotImplementedError

    obj_1 = env.scene[
        obj_1_cfg.name if isinstance(obj_1_cfg, SceneEntityCfg) else obj_1_cfg
    ]
    obj_2 = env.scene[
        obj_2_cfg.name if isinstance(obj_2_cfg, SceneEntityCfg) else obj_2_cfg
    ]
    obj_1_pos = get_pos_w(obj_1, obj_1_cfg)
    obj_2_pos = get_pos_w(obj_2, obj_2_cfg)
    dist = torch.norm(obj_1_pos - obj_2_pos, dim=-1)
    if to_print:
        print(f"Dist between {obj_1_cfg.name} and {obj_2_cfg.name}: {dist}")
    return dist < threshold


def stable_rew(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
):
    robot = env.scene[robot_cfg.name]
    robot_cfg.resolve(env.scene)
    # Get joint velocities
    joint_vel_norm = torch.norm(
        robot.data.joint_vel, dim=-1
    )  # L2 norm of joint velocities
    return torch.clip(joint_vel_norm, 0.0, 5.0)


# def lifting_rew(
#     env: ManagerBasedRLEnv,
#     item_cfg: SceneEntityCfg,
#     robot_cfg: SceneEntityCfg,
#     origin_height: float = 0.42,
#     height_threshold: float = 0.1,
# ):
#     # TODO: should be delta height
#     item_pos = env.scene[item_cfg.name].data.root_pos_w
#     robot = env.scene[robot_cfg.name]
#     robot_cfg.resolve(env.scene)
#     robot_ee_pos = robot.data.body_state_w[:, robot_cfg.body_ids, 0:3].mean(dim=-2)
#     height = item_pos[:, 2:3] - origin_height
#     rew = (torch.tanh(torch.maximum(height_threshold - height, torch.zeros_like(height))) + 1) / 2

#     return rew

# def approaching_rew(
#     env: ManagerBasedRLEnv,
#     item_cfg: SceneEntityCfg,
#     robot_cfg: SceneEntityCfg,
#     threshold: float = 0.1,
# ):
#     # TODO: should be delta distance
#     item_pos = env.scene[item_cfg.name].data.root_pos_w
#     robot = env.scene[robot_cfg.name]
#     robot_cfg.resolve(env.scene)
#     robot_ee_pos = robot.data.body_state_w[:, robot_cfg.body_ids, 0:3].mean(dim=-2)
#     dist = torch.norm(item_pos[:, :2] - robot_ee_pos[:, :2], dim=-1)
#     dist = torch.clamp(dist, min=threshold)
#     rew = (torch.tanh(threshold - dist) + 1) / 2

#     return rew


# TODO: move these two funcs to math_utils
def normalize_vector(x: torch.Tensor, eps=1e-6):
    """normalizes a given torch tensor x and if the norm is less than eps, set the norm to 0"""
    norm = torch.linalg.norm(x, axis=1)
    norm[norm < eps] = 1
    norm = 1 / norm
    return torch.multiply(x, norm[:, None])


def compute_angle_between(x1: torch.Tensor, x2: torch.Tensor):
    """Compute angle (radian) between two torch tensors"""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1, 1)
    return torch.arccos(dot_prod)


def grasp_obs(
    env: ManagerBasedRLEnv,
    robot_grasp_cfg: SceneEntityCfg,
):
    return is_object_grasped(env, robot_grasp_cfg).unsqueeze(-1)


def is_object_grasped(
    env: ManagerBasedRLEnv,
    robot_grasp_cfg: SceneEntityCfg,
    min_force: float = 0.5,
    max_angle: float = 85.0,
):
    robot = env.scene[robot_grasp_cfg.name]
    robot_grasp_cfg.resolve(env.scene)
    left_finger_pose = robot.data.body_state_w[:, robot_grasp_cfg.body_ids[0], 0:7]
    left_finger_matrix = math_utils.matrix_from_quat(left_finger_pose[:, 3:7])
    # gripper open direction
    left_finger_dir = left_finger_matrix[:, :, 0]  # (N, 3)
    left_finger_cf = env.scene[
        robot_grasp_cfg.body_names[0] + "_cf"
    ].data.force_matrix_w[
        :, 0, 0
    ]  # (N, 3)
    langle = compute_angle_between(left_finger_dir, left_finger_cf)

    right_finger_pose = robot.data.body_state_w[:, robot_grasp_cfg.body_ids[1], 0:7]
    right_finger_matrix = math_utils.matrix_from_quat(right_finger_pose[:, 3:7])
    # gripper open direction
    right_finger_dir = -right_finger_matrix[:, :, 0]  # (N, 3)
    right_finger_cf = env.scene[
        robot_grasp_cfg.body_names[1] + "_cf"
    ].data.force_matrix_w[
        :, 0, 0
    ]  # (N, 3)
    rangle = compute_angle_between(right_finger_dir, right_finger_cf)

    lflag = torch.logical_and(
        torch.linalg.norm(left_finger_cf, axis=1) >= min_force,
        torch.rad2deg(langle) <= max_angle,
    )
    rflag = torch.logical_and(
        torch.linalg.norm(right_finger_cf, axis=1) >= min_force,
        torch.rad2deg(rangle) <= max_angle,
    )
    # if (env.sim.current_time_step_index // env.cfg.decimation) % 20 == 0:
    #     print(f"grasped: {torch.logical_and(lflag, rflag)}")
    #     print(f"left finger force, angle: {torch.linalg.norm(left_finger_cf, axis=1)}, {torch.rad2deg(langle)}")
    #     print(f"right finger force, angle: {torch.linalg.norm(right_finger_cf, axis=1)}, {torch.rad2deg(rangle)}")
    return torch.logical_and(lflag, rflag)


def grasped_rew(
    env: ManagerBasedRLEnv,
    robot_grasp_cfg: SceneEntityCfg,
):
    return is_object_grasped(env, robot_grasp_cfg).float()


class lifting_rew(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        self.init_item_pos = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        item_pos = self._env.scene[self.cfg.params["item_cfg"].name].data.root_pos_w
        self.init_item_pos[env_ids] = item_pos[env_ids, :3]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        item_cfg: SceneEntityCfg,
        robot_grasp_cfg: SceneEntityCfg,
        horizontal_threshold: float = 0.1,
        height_threshold: float = 0.1,
    ):
        is_grasped = is_object_grasped(env, robot_grasp_cfg)
        item_pos = env.scene[item_cfg.name].data.root_pos_w
        item_horizontal_movement = torch.norm(
            item_pos[:, :2] - self.init_item_pos[:, :2], dim=-1
        )
        current_heights = item_pos[:, 2:3].squeeze(-1)
        delta_heights = torch.clip(
            current_heights - self.init_item_pos[:, 2:3].squeeze(-1),
            min=0.0,
            max=height_threshold,
        )
        rew = torch.tanh(20.0 * delta_heights)
        rew[~is_grasped] = 0.0
        rew[item_horizontal_movement > horizontal_threshold] = 0.0
        rew[delta_heights > height_threshold] = 0.0
        return rew


class approaching_rew(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.closest_distances = torch.full(
            (env.num_envs,), float("inf"), device=env.device
        )

    def get_pos_w(self, obj, entity_cfg: SceneEntityCfg) -> torch.Tensor:
        if isinstance(obj, Articulation):
            entity_cfg.resolve(self._env.scene)
            return obj.data.body_state_w[:, entity_cfg.body_ids, 0:3].mean(dim=-2)
        elif isinstance(obj, RigidObject):
            return obj.data.root_pos_w
        elif isinstance(obj, FrameTransformer):
            return obj.data.target_pos_w[:, 0, :]
        else:
            raise NotImplementedError

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self.closest_distances[env_ids] = torch.full(
            self.closest_distances[env_ids].shape, float("inf"), device=self._env.device
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        obj_1_cfg: SceneEntityCfg | str,
        obj_2_cfg: SceneEntityCfg | str,
        threshold: float = 0.1,
    ):
        obj_1 = env.scene[
            obj_1_cfg.name if isinstance(obj_1_cfg, SceneEntityCfg) else obj_1_cfg
        ]
        obj_2 = env.scene[
            obj_2_cfg.name if isinstance(obj_2_cfg, SceneEntityCfg) else obj_2_cfg
        ]
        obj_1_pos = self.get_pos_w(obj_1, obj_1_cfg)
        obj_2_pos = self.get_pos_w(obj_2, obj_2_cfg)

        current_distances = torch.norm(obj_1_pos - obj_2_pos, dim=-1)
        delta_distances = torch.where(
            torch.isinf(self.closest_distances),
            torch.zeros_like(self.closest_distances),
            self.closest_distances - current_distances,
        )  # if the sofar closest distance is inf (the first env step), the delta distance should be 0
        self.closest_distances = torch.minimum(
            self.closest_distances, current_distances
        )
        delta_distances = torch.clip(delta_distances, 0.0, 10.0)
        rew = torch.where(
            current_distances < threshold,
            torch.zeros_like(delta_distances),
            torch.tanh(10.0 * delta_distances),
        )
        return rew


def joint_vel_penality(
    env: ManagerBasedRLEnv,
):
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel[:, :6]
    rew = 1 - torch.tanh(2.0 * torch.norm(joint_vel, dim=-1))
    return rew


def joint_acc_penality(
    env: ManagerBasedRLEnv,
):
    robot = env.scene["robot"]
    joint_acc = robot.data.joint_acc[:, :6]
    rew = 1 - torch.tanh(2.0 * torch.norm(joint_acc, dim=-1))
    return rew


def approach_rew(
    env: ManagerBasedRLEnv,
    obj_1_cfg: SceneEntityCfg | str,
    obj_2_cfg: SceneEntityCfg | str,
    threshold: float = 0.0,
):
    def get_pos_w(obj, entity_cfg: SceneEntityCfg) -> torch.Tensor:
        if isinstance(obj, Articulation):
            entity_cfg.resolve(env.scene)
            return obj.data.body_state_w[:, entity_cfg.body_ids, 0:3].mean(dim=-2)
        elif isinstance(obj, RigidObject):
            return obj.data.root_pos_w
        elif isinstance(obj, FrameTransformer):
            return obj.data.target_pos_w[:, 0, :]
        else:
            raise NotImplementedError

    obj_1 = env.scene[
        obj_1_cfg.name if isinstance(obj_1_cfg, SceneEntityCfg) else obj_1_cfg
    ]
    obj_2 = env.scene[
        obj_2_cfg.name if isinstance(obj_2_cfg, SceneEntityCfg) else obj_2_cfg
    ]
    obj_1_pos = get_pos_w(obj_1, obj_1_cfg)
    obj_2_pos = get_pos_w(obj_2, obj_2_cfg)
    dist = torch.norm(obj_1_pos - obj_2_pos, dim=-1)
    rew = 1 - torch.tanh(5 * torch.clamp(dist - threshold, min=0.0))
    # if (env.sim.current_time_step_index // env.cfg.decimation) % 20 == 0:
    #     print(f"dist: {dist}")
    return rew


class horizontal_offset_penality(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.prev_item_xy = torch.full(
            (env.num_envs, 2), float("inf"), device=env.device
        )

    def reset(self, env_ids: torch.Tensor):
        item_pos = self._env.scene[self.cfg.params["item_cfg"].name].data.root_pos_w
        self.prev_item_xy[env_ids] = item_pos[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        item_cfg: SceneEntityCfg,
    ):
        item_pos = env.scene[item_cfg.name].data.root_pos_w
        current_item_xy = item_pos[:, :2]
        item_xy_movement = torch.norm(current_item_xy - self.prev_item_xy, dim=-1)
        rew = 1 - torch.tanh(2.0 * item_xy_movement)
        self.prev_item_xy[:] = current_item_xy[:]
        return rew


class and_func(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: ManagerTermBaseCfg):
        super().__init__(cfg, env)
        for key in self.cfg.params["funcs"]:
            if inspect.isclass(self.cfg.params["funcs"][key]) and issubclass(
                self.cfg.params["funcs"][key], ManagerTermBase
            ):
                cfg_temp = ManagerTermBaseCfg()
                cfg_temp.params = self.cfg.params["kwargs"][key]
                cfg_temp.func = lambda x: 1
                self.cfg.params["funcs"][key] = self.cfg.params["funcs"][key](
                    env, cfg_temp
                )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        for key in self.cfg.params["funcs"]:
            if isinstance(self.cfg.params["funcs"][key], ManagerTermBase):
                self.cfg.params["funcs"][key].reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        funcs: dict[str, callable],
        kwargs: dict[str, dict] = {},
    ) -> torch.Tensor:
        """
        A function that takes a list of functions and their corresponding keyword arguments
        and returns True if all functions return True, otherwise returns False.
        """
        results = []
        for key in funcs:
            results.append(funcs[key](env, **kwargs[key]))
        return reduce(torch.logical_and, results)


def position_range(
    env: ManagerBasedRLEnv,
    item_cfg: SceneEntityCfg,
    threshold: dict = {},
):
    item_pos = env.scene[item_cfg.name].data.root_pos_w - env.scene.env_origins
    item_pos = item_pos[:, :3]
    item_quat = env.scene[item_cfg.name].data.root_quat_w
    ex, ey, ez = math_utils.euler_xyz_from_quat(item_quat)
    res = torch.ones_like(item_pos[:, 0])
    for key in threshold:
        lb, up = threshold[key]
        if lb is None:
            lb = -float("inf")
        if up is None:
            up = float("inf")
        if key == "x":
            in_position = torch.logical_and(item_pos[:, 0] > lb, item_pos[:, 0] < up)
        elif key == "y":
            in_position = torch.logical_and(item_pos[:, 1] > lb, item_pos[:, 1] < up)
        elif key == "z":
            in_position = torch.logical_and(item_pos[:, 2] > lb, item_pos[:, 2] < up)
        else:
            raise ValueError(f"Unknown key {key} in threshold.")
        res = torch.logical_and(res, in_position)
    return res


def near_one_vector(
    env: ManagerBasedRLEnv,
    item_cfg: SceneEntityCfg,
    vec_ref: torch.Tensor,
    local_vec: torch.Tensor,
    lb: float = None,
    ub: float = None,
):
    if lb is None:
        lb = -float("inf")
    if ub is None:
        ub = float("inf")
    item_quat = env.scene[item_cfg.name].data.root_quat_w
    matrix = math_utils.matrix_from_quat(item_quat)
    vec_ref = vec_ref.reshape(1, 3, 1).to(env.device)
    vec_ref = vec_ref / torch.linalg.norm(vec_ref, dim=1, keepdim=True)
    local_vec = local_vec.reshape(1, 3, 1).to(env.device)
    local_vec = local_vec / torch.linalg.norm(local_vec, dim=1, keepdim=True)
    vec_ref_now = matrix @ vec_ref
    inner_product = torch.sum(vec_ref_now * local_vec, dim=(-2, -1))  # (N,)
    return torch.logical_and(inner_product > lb, inner_product < ub)


class a_in_b:
    def __init__(self):
        self.initialized: bool = False
        self.b_equations: torch.Tensor = None
        self.a_points: torch.Tensor = None

    def _initialize(self, env, a: SceneEntityCfg, b: SceneEntityCfg):
        self.initialized = True
        obj_a: RigidObject = env.scene[a.name]
        obj_b: RigidObject | Articulation = env.scene[b.name]
        a_prim_path = str(XFormPrim(obj_a.cfg.prim_path).prims[0].GetPrimPath())
        b_prim_path = str(XFormPrim(obj_b.cfg.prim_path).prims[0].GetPrimPath())
        self.a_points = get_points_at_path(a_prim_path)
        self.a_points = (
            torch.from_numpy(self.a_points).to(env.device).repeat((env.num_envs, 1, 1))
        )
        b_points = get_points_at_path(b_prim_path)
        hull = ConvexHull(b_points)
        self.b_equations = torch.tensor(hull.equations, device=env.device).repeat(
            (env.num_envs, 1, 1)
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        a: SceneEntityCfg,
        b: SceneEntityCfg,
        ratio: float = 1.0,
    ):
        if not self.initialized:
            self._initialize(env, a, b)

        obj_a: RigidObject = env.scene[a.name]
        obj_b: Articulation = env.scene[b.name]

        # Transform points to world space
        a_to_world = math_utils.make_pose(
            obj_a.data.root_pos_w, math_utils.matrix_from_quat(obj_a.data.root_quat_w)
        )
        b_to_world = math_utils.make_pose(
            obj_b.data.root_pos_w, math_utils.matrix_from_quat(obj_b.data.root_quat_w)
        )

        a_points = self.a_points.clone()

        a_points_world = (
            a_to_world
            @ torch.cat([a_points, torch.ones_like(a_points[..., :1])], dim=-1)
            .to(dtype=torch.float32)
            .transpose(-2, -1)
        ).transpose(-2, -1)
        a_points_in_b = (
            torch.linalg.inv(b_to_world) @ a_points_world.transpose(-2, -1)
        ).transpose(-2, -1)

        results = self.b_equations @ a_points_in_b.transpose(-2, -1).to(
            self.b_equations
        )

        if ratio >= 0.999:
            inside = (results <= 0).all(dim=[-1, -2])
        else:
            # inside = (results <= 0).sum(dim=[-1, -2]) / results.shape[1] >= ratio
            inside = (results <= 0).all(dim=-2).sum(-1) / results.shape[2] >= ratio

        return inside


def get_camera_image(
    env: ManagerBasedRLEnv,
    camera_entity: SceneEntityCfg,
    return_on_cpu: bool = True,
):
    camera: Camera = env.scene[camera_entity.name]
    if return_on_cpu:
        return camera.data.output["rgb"].cpu()
    else:
        return camera.data.output["rgb"]


def get_camera_depth(
    env: ManagerBasedRLEnv,
    camera_entity: SceneEntityCfg,
    return_on_cpu: bool = True,
    resize: bool = True,
    resize_shape: Tuple[int] = None,  # (224, 224)
):
    camera: Camera = env.scene[camera_entity.name]
    depth = camera.data.output["depth"]

    if resize and resize_shape is not None:
        # Move to CPU for cv2 processing
        import cv2

        depth_cpu = depth.cpu() if depth.is_cuda else depth
        depth_np = depth_cpu.numpy()

        # Resize each image in the batch
        batch_size = depth_np.shape[0]
        resized_images = []
        for i in range(batch_size):
            resized = cv2.resize(
                depth_np[i],
                (resize_shape[1], resize_shape[0]),  # cv2 expects (W, H)
                interpolation=cv2.INTER_NEAREST,
            )
            resized_images.append(resized)

        # Convert back to tensor
        depth = torch.from_numpy(np.stack(resized_images))

        # Move back to original device if needed
        if not return_on_cpu and camera.data.output["depth"].is_cuda:
            depth = depth.cuda()
    else:
        if return_on_cpu:
            depth = depth.cpu()

    # Add channel dimension to ensure output shape is (B, H, W, 1)
    if len(depth.shape) == 3:  # (B, H, W)
        depth = depth.unsqueeze(-1)  # (B, H, W, 1)

    return depth


def get_camera_intr(env: ManagerBasedRLEnv, camera_entity: SceneEntityCfg):
    camera: Camera = env.scene[camera_entity.name]
    return camera.data.intrinsic_matrices


def get_camera_extr(env: ManagerBasedRLEnv, camera_entity: SceneEntityCfg):
    camera: Camera = env.scene[camera_entity.name]
    return torch.cat([camera.data.pos_w, camera.data.quat_w_ros], dim=-1)


def get_pointcloud_obs(
    env: ManagerBasedRLEnv,
    cameras: List[SceneEntityCfg],
    robot_base: bool = True,
    crop_bound=[0.2, 1.5, -1.2, 1.2, -0.3, 0.7],
    npoints=8192,
    return_on_cpu: bool = True,
):
    # assert the output is on cpu
    pcds = []
    colors = []
    if robot_base:
        robot: Articulation = env.scene["robot"]
    for i in range(len(cameras)):
        camera: Camera = env.scene[cameras[i].name]
        if robot_base:
            position, orientation = math_utils.subtract_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                camera.data.pos_w,
                camera.data.quat_w_ros,
            )
        else:
            position = torch.zeros_like(camera.data.pos_w)
            orientation = torch.zeros_like(camera.data.quat_w_ros)
            orientation[..., 0] = 1.0  # identity quaternion
        res = create_pointcloud_from_rgbd_batch(
            intrinsic_matrix=camera.data.intrinsic_matrices,
            depth=camera.data.output["depth"][..., 0],
            rgb=camera.data.output["rgb"],
            position=position,
            orientation=orientation,
        )
        pcds.append(res["pos"])
        colors.append(res["color"])
    pos = torch.concatenate(pcds, dim=1)
    color = torch.concatenate(colors, dim=1)
    del pcds, colors
    obs = pcd_downsample_torch(
        dict(pos=pos, color=color),
        bound=crop_bound,
        bound_clip=(crop_bound is not None),
        num=npoints,
    )
    if return_on_cpu:
        pos = obs["pos"].cpu()
        color = obs["color"].cpu()
    return torch.concatenate([pos[..., None], color[..., None]], dim=-1)


def ee_frame_pos(
    env: ManagerBasedRLEnv,
    entity_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
) -> torch.Tensor:
    robot = env.scene[entity_cfg.name]
    entity_cfg.resolve(env.scene)
    ee_id = entity_cfg.body_ids[0]

    ee_pose_w = robot.data.body_state_w[:, ee_id, 0:7]
    root_pose_w = robot.data.body_state_w[:, 0, 0:7]
    ee_pose_b = torch.cat(
        math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        ),
        dim=1,
    )
    return ee_pose_b[:, :3]


def ee_frame_orn(
    env: ManagerBasedRLEnv,
    entity_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
    type="quat",
) -> torch.Tensor:
    robot = env.scene[entity_cfg.name]
    entity_cfg.resolve(env.scene)
    ee_id = entity_cfg.body_ids[0]

    ee_pose_w = robot.data.body_state_w[:, ee_id, 0:7]
    root_pose_w = robot.data.body_state_w[:, 0, 0:7]
    ee_pose_to_base = torch.cat(
        math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        ),
        dim=1,
    )
    if type == "quat":
        return math_utils.quat_unique(ee_pose_to_base[:, 3:])
    elif type == "euler":
        raise NotImplementedError
        return math_utils.quat_to_euler(ee_pose_to_base[:, 3:])


def close_articulation(
    env: ManagerBasedRLEnv,
    item_entity: SceneEntityCfg,
    threshold: tuple[float, float] = (-0.1, 0.1),
):

    item: Articulation = env.scene[item_entity.name]
    item_entity.resolve(env.scene)
    joint = item.data.joint_pos[:, item_entity.joint_ids]  # (env, num_joint)
    return torch.logical_and(joint > threshold[0], joint < threshold[1]).all(dim=-1)


def _get_eef_subtask_term_signal_idx(_subtask_signal_to_eef_and_idx, eef_name):
    index = 0
    for subtask_signal in _subtask_signal_to_eef_and_idx:
        for eef_name_this in _subtask_signal_to_eef_and_idx[subtask_signal]:
            if eef_name == eef_name_this:
                index += len(
                    _subtask_signal_to_eef_and_idx[subtask_signal][eef_name_this]
                )
    return index


class success_all_task_sequentially(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg, debug=False):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self._subtask_history = {}
        self._subtask_signal_to_eef_and_idx: Dict[str, Dict[str, List[int]]] = {}
        for eef_name in self._env.cfg.subtask_configs:
            subtask_cfgs: List["WbcSubTaskConfig"] = self._env.cfg.subtask_configs[
                eef_name
            ]
            self._subtask_history[eef_name] = torch.zeros(
                (env.num_envs, len(subtask_cfgs)), device=env.device, dtype=torch.int
            )
            for subtask_cfg in subtask_cfgs:
                if (
                    subtask_cfg.subtask_term_signal
                    not in self._subtask_signal_to_eef_and_idx
                ):
                    self._subtask_signal_to_eef_and_idx[
                        subtask_cfg.subtask_term_signal
                    ] = {}
                if (
                    eef_name
                    not in self._subtask_signal_to_eef_and_idx[
                        subtask_cfg.subtask_term_signal
                    ]
                ):
                    self._subtask_signal_to_eef_and_idx[
                        subtask_cfg.subtask_term_signal
                    ][eef_name] = []

                self._subtask_signal_to_eef_and_idx[subtask_cfg.subtask_term_signal][
                    eef_name
                ].append(
                    _get_eef_subtask_term_signal_idx(
                        self._subtask_signal_to_eef_and_idx, eef_name
                    )
                )
        self._debug = debug

    def reset(self, env_ids: torch.Tensor = None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs)
        for eef_name in self._subtask_history:
            self._subtask_history[eef_name][env_ids, :] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        env_one_hots = torch.zeros(
            (self._env.num_envs, 1), device=self._env.device, dtype=torch.bool
        )
        env_one_hots[env_ids] = True

        for subtask_term_signal in self._subtask_signal_to_eef_and_idx:
            for eef_name in self._subtask_signal_to_eef_and_idx[subtask_term_signal]:
                idxes = self._subtask_signal_to_eef_and_idx[subtask_term_signal][
                    eef_name
                ]
                self._subtask_history[eef_name][:, idxes] += (
                    self._env.obs_buf["subtask_terms"][subtask_term_signal].reshape(
                        -1, 1
                    )
                    * env_one_hots
                )

        # check if all subtasks are finished
        all_subtasks_finished = torch.ones(
            self._env.num_envs, device=self._env.device, dtype=torch.bool
        )
        for eef_name in self._subtask_history:
            all_subtasks_finished = torch.logical_and(
                all_subtasks_finished, self._subtask_history[eef_name].all(dim=-1)
            )
        if self._debug:
            for eef_name in self._subtask_history:
                print(
                    f"eef_name: {eef_name}, subtask history: \n{self._subtask_history[eef_name]}"
                )
        return all_subtasks_finished[env_ids]


def always_true(
    env: ManagerBasedRLEnv,
):
    """A dummy function that always returns True."""
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


def always_false(
    env: ManagerBasedRLEnv,
):
    """A dummy function that always returns False."""
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def reset_camera_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Camera | TiledCamera = env.scene[asset_cfg.name]
    # get default root state
    root_states_pos = asset.data.pos_w[env_ids].clone()
    root_states_quat = asset.data.quat_w_ros[env_ids].clone()

    # poses
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    positions = root_states_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states_quat, orientations_delta)
    # set into the physics simulation
    asset.set_world_poses(
        positions=positions, orientations=orientations, env_ids=env_ids.cpu().tolist()
    )


def robot_joint_pos_success(
    env: ManagerBasedRLEnv,
    robot_entity: SceneEntityCfg,
    threshold: tuple[float, float] = (-0.1, 0.1),
):
    """Check if the robot's joint positions are within the success threshold."""
    robot: Articulation = env.scene[robot_entity.name]
    robot_entity.resolve(env.scene)
    joint_positions = robot.data.joint_pos[:, robot_entity.joint_ids]  # (
    return (
        (joint_positions >= threshold[0]) & (joint_positions <= threshold[1])
    ).reshape(-1)


def one_hot_gripper_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper_joint"]),
    threshold: float = 0.1,
    reverse: bool = False,
) -> torch.Tensor:
    asset_cfg.resolve(env.scene)
    robot: Articulation = env.scene[asset_cfg.name]
    gripper_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if not reverse:
        one_hot_gripper_pos = torch.where(
            gripper_pos > threshold,
            torch.ones_like(gripper_pos),
            torch.zeros_like(gripper_pos),
        )
    else:
        one_hot_gripper_pos = torch.where(
            gripper_pos < threshold,
            torch.ones_like(gripper_pos),
            torch.zeros_like(gripper_pos),
        )
    return one_hot_gripper_pos
