# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import einops
import isaaclab.utils.math as MathUtils

# from ..envs import
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_mimic.utils.prim import get_points_at_path
from isaacsim.core.prims import XFormPrim
from scipy.spatial import ConvexHull

from .....mdp import *


class pour_success:
    def __init__(self):
        self.initialized = False
        self.pick_item_points = None
        self.place_item_points = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        pick_item_cfg: SceneEntityCfg,
        place_item_cfg: SceneEntityCfg,
        threshold: float = 0.01,
        origin_height: float = 0.42,
        height_offset: float = 0.05,
        pick_item_z: int = 1,
        place_item_z: int = 1,
    ):
        pick_item: RigidObject = env.scene[pick_item_cfg.name]
        place_item: RigidObject = env.scene[place_item_cfg.name]
        if not self.initialized:
            pick_item_prim_path = str(
                XFormPrim(pick_item.cfg.prim_path).prims[0].GetPrimPath()
            )
            place_item_prim_path = str(
                XFormPrim(place_item.cfg.prim_path).prims[0].GetPrimPath()
            )
            self.pick_item_points = torch.from_numpy(
                get_points_at_path(pick_item_prim_path)
            ).to(env.device)
            self.place_item_points = torch.from_numpy(
                get_points_at_path(place_item_prim_path)
            ).to(env.device)
            self.initialized = True
            self.max_pick_item_points = torch.max(
                self.pick_item_points.abs(), dim=0
            ).values
            self.pick_item_focus = torch.tensor([0.0, 0.0, 0.0]).to(env.device)
            self.pick_item_focus[pick_item_z] = self.max_pick_item_points[pick_item_z]
            self.pick_item_focus = torch.cat(
                [self.pick_item_focus, torch.tensor([1], device=env.device)]
            )[:, None].repeat(
                (env.num_envs, 1, 1)
            )  # (env.num_envs, 4, 1)

            self.idxes = [0, 1, 2]
            self.idxes.remove(place_item_z)
            place_hull = ConvexHull(self.place_item_points[:, self.idxes].cpu().numpy())
            self.place_eq = (
                torch.tensor(place_hull.equations[:, :-1], dtype=torch.float)
                .to(env.device)
                .repeat((env.num_envs, 1, 1))
            )  # (env.num_envs, convex_hull.num_equations, 2)
            self.place_c = (
                torch.tensor(place_hull.equations[:, -1], dtype=torch.float)
                .to(env.device)
                .repeat((env.num_envs, 1))
            )  # (env.num_envs, convex_hull.num_equations)
            self.in_place_convex_hull = (
                lambda points: (
                    self.place_eq[..., None, :]
                    @ points[:, None, :, :].repeat((1, self.place_eq.shape[1], 1, 1))
                    + self.place_c
                )[..., 0, 0]
                <= 0.0
            )

        pick_item_to_place_item = torch.cat(
            MathUtils.subtract_frame_transforms(
                place_item.data.root_pos_w,
                place_item.data.root_quat_w,
                pick_item.data.root_pos_w,
                pick_item.data.root_quat_w,
            ),
            dim=-1,
        ).to(env.device)
        pick_item_to_place_item_pose = MathUtils.make_pose(
            pick_item_to_place_item[:, :3],
            MathUtils.matrix_from_quat(pick_item_to_place_item[:, 3:]),
        )  # (env.num_envs, 4, 4)
        pick_focus_in_place_frame = (
            pick_item_to_place_item_pose @ self.pick_item_focus
        )[:, self.idxes, :]
        return self.in_place_convex_hull(pick_focus_in_place_frame).all(dim=-1)


class line_objects:
    def __init__(self):
        self.initialized = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        objects_enties: list[SceneEntityCfg],
        threshold=0.9,
    ):
        if not self.initialized:
            self.initialized = True
            self.object_means = {}
            for object_cfg in objects_enties:
                object: RigidObject = env.scene[object_cfg.name]
                object_prim_path = str(
                    XFormPrim(object.cfg.prim_path).prims[0].GetPrimPath()
                )
                object_points = torch.from_numpy(
                    get_points_at_path(object_prim_path)
                ).to(env.device)
                self.object_means[object_cfg.name] = torch.cat(
                    [
                        torch.mean(object_points, dim=0),
                        torch.tensor([1], device=env.device),
                    ]
                )[:, None].repeat(
                    (env.num_envs, 1, 1)
                )  # env.num_envs, 4, 1

        object_mean_in_worlds = []
        for object_cfg in objects_enties:
            object: RigidObject = env.scene[object_cfg.name]
            object_to_world = MathUtils.make_pose(
                object.data.root_pos_w,
                MathUtils.matrix_from_quat(object.data.root_quat_w),
            )  # env.num_envs, 4, 4
            object_mean_in_world = (
                object_to_world @ self.object_means[object_cfg.name]
            )  # env.num_envs, 4, 1
            object_mean_in_worlds.append(object_mean_in_world)

        return torch.tensor([False] * env.num_envs, device=env.device)


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
