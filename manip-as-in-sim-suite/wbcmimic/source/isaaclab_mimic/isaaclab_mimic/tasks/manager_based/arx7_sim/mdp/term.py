# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def approach_success(
    env: ManagerBasedRLEnv,
    item_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    threshold: float = 0.01,
):
    item_pos = env.scene[item_cfg.name].data.root_pos_w
    robot = env.scene[robot_cfg.name]
    robot_cfg.resolve(env.scene)
    robot_ee_pos = robot.data.body_state_w[:, robot_cfg.body_ids, 0:3].mean(dim=-2)
    dist = torch.norm(item_pos[:, :2] - robot_ee_pos[:, :2], dim=-1)
    return dist < threshold


def lift_success(
    env: ManagerBasedRLEnv,
    item_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    threshold: float = 0.01,
    origin_height: float = 0.42,
    height_threshold: float = 0.1,
):
    item_pos = env.scene[item_cfg.name].data.root_pos_w
    robot = env.scene[robot_cfg.name]
    robot_cfg.resolve(env.scene)
    robot_ee_pos = robot.data.body_state_w[:, robot_cfg.body_ids, 0:3].mean(dim=-2)
    dist = torch.norm(item_pos[:, :2] - robot_ee_pos[:, :2], dim=-1)[:, None]
    return torch.logical_and(
        dist < threshold, item_pos[:, 2:3] > origin_height + height_threshold
    )
