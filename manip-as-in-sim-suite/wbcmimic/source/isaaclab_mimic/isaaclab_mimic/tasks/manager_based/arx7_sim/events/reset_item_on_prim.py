# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Dict, List

import isaaclab.sim as sim_utils
import isaaclab_mimic.utils.prim as prim_utils
import numpy as np
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def check_overlap_by_pcd_bbox(
    pcd: np.ndarray, table_pcd: np.ndarray, placed_items_pcd: List[np.ndarray]
) -> bool:
    """
    通过比较点云的边界框来检查是否存在重叠

    参数:
        pcd: 要检查的点云
        table_pcd: 桌面的点云
        placed_items_pcd: 已放置物品的点云列表
        threshold: 重叠判定的阈值

    返回:
        bool: 如果重叠返回True，否则返回False
    """
    # 计算要检查的点云的边界框
    pcd_min = np.min(pcd, axis=0)
    pcd_max = np.max(pcd, axis=0)

    # 检查与已放置物品的重叠
    for placed_pcd in placed_items_pcd:
        placed_min = np.min(placed_pcd, axis=0)
        placed_max = np.max(placed_pcd, axis=0)

        # 检查x、y、z三个维度是否都有重叠
        overlap = all(
            pcd_min[i] <= placed_max[i] and pcd_max[i] >= placed_min[i]
            for i in range(3)
        )

        if overlap:
            return True

    return False


def _safe_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    else:
        return t


def _A_To_B(t_ab, q_ab, t_bc, q_bc, torch_output=True, device=None):
    """
    To get t_ac, q_ac
    """
    t_ab = _safe_to_numpy(t_ab)
    q_ab = _safe_to_numpy(q_ab)
    t_bc = _safe_to_numpy(t_bc)
    q_bc = _safe_to_numpy(q_bc)

    num_envs = t_ab.shape[0]

    transform_ab = np.tile(np.eye(4), (num_envs, 1, 1))
    transform_ab[:, :3, :3] = R.from_quat(q_ab, scalar_first=True).as_matrix()
    transform_ab[:, :3, 3] = t_ab

    transform_bc = np.tile(np.eye(4), (num_envs, 1, 1))
    transform_bc[:, :3, :3] = R.from_quat(q_bc, scalar_first=True).as_matrix()
    transform_bc[:, :3, 3] = t_bc

    transform_ac = transform_bc @ transform_ab
    t_ac = transform_ac[:, :3, 3]
    q_ac = R.from_matrix(transform_ac[:, :3, :3]).as_quat(scalar_first=True)

    if torch_output:
        t_ac = torch.tensor(t_ac, device=device)
        q_ac = torch.tensor(q_ac, device=device)
    else:
        t_ac = t_ac
        q_ac = q_ac
    return t_ac, q_ac


class ResetItemOnPrim:
    def __init__(self):
        self.id_name_to_pcd: Dict[int, Dict[str, np.ndarray]] = {}
        self.id_name_to_pcd_min: Dict[int, Dict[str, np.ndarray]] = {}
        self.id_name_to_pcd_max: Dict[int, Dict[str, np.ndarray]] = {}
        self.initialized = False
        self.id_name_to_item_origin_xy: Dict[int, Dict[str, torch.Tensor]] = {}

    def initialize(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        random_item_cfgs: List[SceneEntityCfg],
        place_prim_cfg: SceneEntityCfg,
    ):
        # for item_cfg in random_item_cfgs:
        #     item = env.scene[item_cfg.name]
        # item_pose = item.get_local_poses()
        # for i in range(env.num_envs):
        #     if item._prims[i].GetAttribute("xformOp:orient") not in item._prims[i].GetAttributes():
        #         xformable = UsdGeom.Xformable(item._prims[i])
        #         xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
        #         item._set_xform_properties()
        # item.set_local_poses(item_pose[0], item_pose[1])

        self.initialized = True
        for env_id in env_ids:
            self.id_name_to_pcd[int(env_id.item())] = {}
            self.id_name_to_pcd_min[int(env_id.item())] = {}
            self.id_name_to_pcd_max[int(env_id.item())] = {}
            self.id_name_to_item_origin_xy[int(env_id.item())] = {}
            for item_cfg in random_item_cfgs:
                item: RigidObject = env.scene[item_cfg.name]
                self.id_name_to_pcd[int(env_id.item())][item_cfg.name] = (
                    prim_utils.get_points_at_path(
                        sim_utils.find_matching_prim_paths(item.cfg.prim_path)[0]
                    )
                )
                self.id_name_to_pcd_min[int(env_id.item())][item_cfg.name] = np.min(
                    self.id_name_to_pcd[int(env_id.item())][item_cfg.name], axis=0
                )
                self.id_name_to_pcd_max[int(env_id.item())][item_cfg.name] = np.max(
                    self.id_name_to_pcd[int(env_id.item())][item_cfg.name], axis=0
                )
                self.id_name_to_item_origin_xy[int(env_id.item())][item_cfg.name] = (
                    item.data.root_pos_w[:2]
                )
            prim = env.scene[place_prim_cfg.name]
            self.id_name_to_pcd[int(env_id.item())][place_prim_cfg.name] = (
                prim_utils.get_points_at_path(prim.prim_paths[0])
            )
            self.id_name_to_pcd_min[int(env_id.item())][place_prim_cfg.name] = np.min(
                self.id_name_to_pcd[int(env_id.item())][place_prim_cfg.name], axis=0
            )
            self.id_name_to_pcd_max[int(env_id.item())][place_prim_cfg.name] = np.max(
                self.id_name_to_pcd[int(env_id.item())][place_prim_cfg.name], axis=0
            )

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        random_item_cfgs: List[SceneEntityCfg],
        place_prim_cfg: SceneEntityCfg,
        rotation_range: dict[str, tuple[float, float]],
        height_offset: float = 0.02,
        fix_height: float = None,
        use_prim_max_height: bool = False,
        max_attemp_times: int = 20,
    ):
        """
        random_item_cfgs: The list of item configurations to be randomized
        place_prim_cfg: The configuration of the place item's prim (e.g. the table)
        rotation_range: The range of the rotation
        height_offset: The offset of the height
        use_prim_max_height: Whether to use the max height of the primitive
        max_attemp_times: The maximum number of attempts
        """
        if not self.initialized:
            self.initialize(env, env_ids, random_item_cfgs, place_prim_cfg)

        final_positions = {}
        final_orientations = {}

        for env_id in env_ids:
            placed_items_pcd = []
            item_positions = {}
            item_orientations = {}
            prim_pcd = self.id_name_to_pcd[int(env_id.item())][place_prim_cfg.name]
            prim = env.scene[place_prim_cfg.name]
            x_range = (
                self.id_name_to_pcd_min[int(env_id.item())][place_prim_cfg.name][0],
                self.id_name_to_pcd_max[int(env_id.item())][place_prim_cfg.name][0],
            )
            y_range = (
                self.id_name_to_pcd_min[int(env_id.item())][place_prim_cfg.name][1],
                self.id_name_to_pcd_max[int(env_id.item())][place_prim_cfg.name][1],
            )

            for item_cfg in random_item_cfgs:
                item_pcd = self.id_name_to_pcd[int(env_id.item())][item_cfg.name]
                item_min = self.id_name_to_pcd_min[int(env_id.item())][item_cfg.name]
                item_max = self.id_name_to_pcd_max[int(env_id.item())][item_cfg.name]

                for _ in range(max_attemp_times):
                    range_list = [
                        rotation_range.get(key, (0.0, 0.0))
                        for key in ["roll", "pitch", "yaw"]
                    ]
                    roll = np.random.uniform(range_list[0][0], range_list[0][1])
                    pitch = np.random.uniform(range_list[1][0], range_list[1][1])
                    yaw = np.random.uniform(range_list[2][0], range_list[2][1])
                    rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=True)

                    item_pcd_mean = np.mean(item_pcd, axis=0)
                    # 计算旋转后的物品尺寸
                    rotated_item = rotation.apply(item_pcd - item_pcd_mean)
                    rotated_min = np.min(rotated_item, axis=0)
                    rotated_max = np.max(rotated_item, axis=0)
                    rotated_size = rotated_max - rotated_min
                    x = (
                        np.random.uniform(
                            x_range[0] - rotated_min[0], x_range[1] - rotated_max[0]
                        )
                        - item_pcd_mean[0]
                    )
                    y = (
                        np.random.uniform(
                            y_range[0] - rotated_min[1], y_range[1] - rotated_max[1]
                        )
                        - item_pcd_mean[1]
                    )

                    if fix_height is not None:
                        z = fix_height
                    else:
                        # 计算物品正下方的桌面高度
                        item_center_xy = np.array([x, y])
                        xy_min = rotated_min[:2] + item_center_xy + item_pcd_mean[:2]
                        xy_max = rotated_max[:2] + item_center_xy + item_pcd_mean[:2]

                        table_pcd_in_xy_bool = (
                            (prim_pcd[:, 0] > xy_min[0])
                            & (prim_pcd[:, 0] < xy_max[0])
                            & (prim_pcd[:, 1] > xy_min[1])
                            & (prim_pcd[:, 1] < xy_max[1])
                        )
                        table_pcd_in_xy = prim_pcd[table_pcd_in_xy_bool]

                        if len(table_pcd_in_xy) <= 0 or use_prim_max_height:
                            z = (
                                np.max(prim_pcd[:, 2])
                                + height_offset
                                - rotated_min[2]
                                - item_pcd_mean[2]
                            )
                        else:
                            table_pcd_in_xy_max_height = np.max(table_pcd_in_xy[:, 2])
                            z = (
                                table_pcd_in_xy_max_height
                                + height_offset
                                - rotated_min[2]
                                - item_pcd_mean[2]
                            )

                    # 创建变换矩阵
                    transform = np.eye(4)
                    transform[:3, :3] = rotation.as_matrix()
                    transform[:3, 3] = [x, y, z]

                    # 变换物品点云
                    homogeneous_pcd = np.hstack(
                        (item_pcd, np.ones((item_pcd.shape[0], 1)))
                    )
                    transformed_pcd = (transform @ homogeneous_pcd.T).T[:, :3]

                    # 检查是否与桌面或其他物品重叠
                    if not check_overlap_by_pcd_bbox(
                        transformed_pcd, prim_pcd, placed_items_pcd
                    ):
                        item_positions[item_cfg.name] = [x, y, z]
                        item_orientations[item_cfg.name] = rotation.as_quat(
                            scalar_first=True
                        )
                        placed_items_pcd.append(transformed_pcd)
                        break

            if not final_positions:
                for item_cfg in random_item_cfgs:
                    final_positions[item_cfg.name] = torch.tensor(
                        [item_positions[item_cfg.name]], device=env.device
                    )
                    final_orientations[item_cfg.name] = torch.tensor(
                        [item_orientations[item_cfg.name]], device=env.device
                    )
            else:
                for item_cfg in random_item_cfgs:
                    final_positions[item_cfg.name] = torch.cat(
                        [
                            final_positions[item_cfg.name],
                            torch.tensor(
                                [item_positions[item_cfg.name]], device=env.device
                            ),
                        ],
                        dim=0,
                    )
                    final_orientations[item_cfg.name] = torch.cat(
                        [
                            final_orientations[item_cfg.name],
                            torch.tensor(
                                [item_orientations[item_cfg.name]], device=env.device
                            ),
                        ],
                        dim=0,
                    )

        for item_cfg in random_item_cfgs:
            item: RigidObject = env.scene[item_cfg.name]
            # for i, prim in enumerate(item.prims):
            #     euler_attr = prim.GetAttribute("xformOp:rotateXYZ")
            #     trans_attr = prim.GetAttribute("xformOp:translate")
            #     euler_degree = R.from_quat(final_orientations[item_cfg.name][i], scalar_first=True).as_euler('xyz', degrees=True)
            #     euler_attr.Set(Gf.Vec3f(euler_degree[0], euler_degree[1], euler_degree[2]))
            #     trans_attr.Set(Gf.Vec3f(final_positions[item_cfg.name][i][0], final_positions[item_cfg.name][i][1], final_positions[item_cfg.name][i][2]))
            #     prim.GetAttribute('physics:angularVelocity').Set(Gf.Vec3f(0.0, 0.0, 0.0))
            #     prim.GetAttribute('physics:velocity').Set(Gf.Vec3f(0.0, 0.0, 0.0))

            # item.set_local_poses(final_positions[item_cfg.name] + env.scene.env_origins[env_ids], final_orientations[item_cfg.name])
            # item.set_local_velocities(torch.zeros((final_positions[item_cfg.name].shape[0], 3), device=env.device), env_ids=env_ids)
            prim_pos, prim_quat = prim.get_local_poses()
            position, orientation = _A_To_B(
                final_positions[item_cfg.name],
                final_orientations[item_cfg.name],
                prim_pos + env.scene.env_origins[env_ids],
                prim_quat,
                torch_output=True,
                device=env.device,
            )
            item_state = item.data.default_root_state.clone()
            # item_state[:, :2] = position[:, :2] + self.id_name_to_item_origin_xy[int(env_ids.item())][item_cfg.name][:, :2]
            item_state[:, :2] = position[:, :2]
            item_state[:, 2] = position[:, 2]
            item_state[:, 3:7] = orientation
            item_state[:, 7:] = 0.0
            item.write_root_pose_to_sim(item_state[:, :7], env_ids=env_ids)
            item.write_root_velocity_to_sim(item_state[:, 7:], env_ids=env_ids)


# 两个坐标系旋转
