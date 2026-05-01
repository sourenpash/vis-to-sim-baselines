# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

import einops
import isaaclab.utils.math as MathUtils

# from ..envs import
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab_mimic.utils.pcd_utils import (
    create_pointcloud_from_rgbd_batch,
    pcd_downsample_torch,
)
from isaaclab_mimic.utils.prim import get_points_at_path
from isaacsim.core.prims import XFormPrim
from scipy.spatial import ConvexHull

from .....mdp import *


def approch_microwave_door(
    env: ManagerBasedRLEnv,
    item_entity: SceneEntityCfg = SceneEntityCfg(
        "microwave", body_names=["Microwave131_Door001"]
    ),
    threshold: tuple[float, float] = (-0.1, 0.1),
):
    item: Articulation = env.scene[item_entity.name]
    item_entity.resolve(env.scene)


def reset_plate_chicken_fork(
    env: ManagerBasedRLEnv,
    env_ids: List[int],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    chicken_cfg: SceneEntityCfg = SceneEntityCfg("chicken"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    plate_pose_range: dict = {},
):
    left_right_range_max = 0.03
    left_right_range_min = 0.0
    fork_right = torch.rand(1).item() < 0.5
    plate: RigidObject = env.scene[plate_cfg.name]
    # chicken: RigidObject = env.scene[chicken_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    plate_range_list = [
        plate_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]
    ]
    ranges = torch.tensor(plate_range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device
    )
    plate_pos = plate
    root_states = plate.data.default_root_state[env_ids].clone()
    plate_positions = (
        root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    )
    plate_orientations = root_states[:, 3:7].clone()
    plate.write_root_pose_to_sim(
        torch.cat([plate_positions, plate_orientations], dim=-1), env_ids=env_ids
    )
    plate.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )

    if fork_right:
        fork_pos_y = (
            torch.rand(len(env_ids), device=env.device)
            * (left_right_range_max - left_right_range_min)
            + left_right_range_min
        ) * -1
        fork_pos = plate_positions + torch.tensor([0.03, 0.0, 0.1], device=env.device)
        fork_pos[:, 1] += fork_pos_y
        # chicken_pos_y = torch.rand(len(env_ids), device=env.device) * (left_right_range_max - left_right_range_min) + left_right_range_min
        # chicken_pos = plate_positions + torch.tensor([0.0, chicken_pos_y, 0.05], device=env.device)
    else:
        fork_pos_y = (
            torch.rand(len(env_ids), device=env.device)
            * (left_right_range_max - left_right_range_min)
            + left_right_range_min
        )
        fork_pos = plate_positions + torch.tensor([0.03, 0.0, 0.1], device=env.device)
        fork_pos[:, 1] += fork_pos_y
        # chicken_pos_y = (torch.rand(len(env_ids), device=env.device) * (left_right_range_max - left_right_range_min) + left_right_range_min) * -1
        # chicken_pos = plate_positions + torch.tensor([0.0, chicken_pos_y, 0.05], device=env.device)
    fork.write_root_pose_to_sim(
        torch.cat(
            [fork_pos, fork.data.default_root_state[env_ids][:, 3:7].clone()], dim=-1
        ),
        env_ids=env_ids,
    )
    fork.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )
    # chicken.write_root_pose_to_sim(torch.cat([chicken_pos, chicken.data.default_root_state[env_ids][:, 3:7].clone()], dim=-1), env_ids=env_ids)


def plate_on_the_table(
    env: ManagerBasedRLEnv,
    plate_entity: SceneEntityCfg = SceneEntityCfg("plate"),
    height: float = 0.01,
    threshold: float = 0.02,
):
    plate: RigidObject = env.scene[plate_entity.name]
    plate_entity.resolve(env.scene)
    plate_height = plate.data.root_pos_w[:, 2].clone()

    return plate_height < height + threshold


if __name__ == "__main__":
    import numpy as np
    import open3d as o3d

    # 创建一个坐标系框架
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]  # 坐标系大小  # 坐标系原点位置
    )

    points = np.ones((1000, 3))
    point_o3d = o3d.geometry.PointCloud()
    point_o3d.points = o3d.utility.Vector3dVector(points)
    # 可视化
    o3d.visualization.draw_geometries([frame_base, point_o3d])
