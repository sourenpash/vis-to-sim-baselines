# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_mimic.utils.path import PROJECT_ROOT

from . import mdp


@configclass
class WashTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the table scene environment."""

    usd_path: str = (
        f"{PROJECT_ROOT}/source/arxx7_assets/Collected_Room_empty_all_table/Room_empty_table.usdc"
    )

    robot: ArticulationCfg = MISSING

    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=UsdFileCfg(
            usd_path=usd_path,
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene/VanityCabinet001",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.96358, -2.52374, 0.00724),
            rot=(0.7071068, 0, 0.0, 0.7071068),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/VanityCabinet001/VanityCabinet001.usda",
        ),
    )
    mirror = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene/HangingCabinet002",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.89975, -2.8049, 1.33982),
            rot=(0.7071068, 0, 0.0, 0.7071068),
        ),
        spawn=UsdFileCfg(
            usd_path=f"{PROJECT_ROOT}/source/arxx7_assets/HangingCabinet002/HangingCabinet002.usda",
        ),
    )


@configclass
class EventsCfg:

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                # "x": (-0.01, 0.01),
                # "y": (-0.01, 0.01),
                # "z": (-0.01, 0.01),
                # "yaw": (-0.004, 0.004),
                # "roll": (-0.004, 0.004),
                # "pitch_yaw": (-0.004, 0.004),
            },
            "velocity_range": {},
        },
    )
