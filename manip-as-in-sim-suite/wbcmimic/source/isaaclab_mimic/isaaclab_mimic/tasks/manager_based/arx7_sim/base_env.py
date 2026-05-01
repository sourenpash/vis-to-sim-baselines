# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .robots.arx_x7 import Arx_X7_URDF_CFG, X7ActionsAllCfg, X7ObservationsCfg


##
# Scene definition
##
@configclass
class UsdAsSceneCfg(InteractiveSceneCfg):

    usd_path: str = MISSING

    robot: ArticulationCfg = MISSING

    def __post_init__(self) -> None:

        self.background = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
            ),
        )
        del self.usd_path


@configclass
class Re3simIlabSim2realSceneCfg(UsdAsSceneCfg):
    """Configuration for a cart-pole scene."""

    # robot
    robot: ArticulationCfg = Arx_X7_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# MDP settings
##


@configclass
class X7LiftRandomEventCfg:
    """Configuration for events."""

    # reset
    reset_lift = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            "position_range": (0.02, 0.3),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class _X7AliveRewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class _TimeoutTerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class Re3simIlabSim2realEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Re3simIlabSim2realSceneCfg = Re3simIlabSim2realSceneCfg(
        num_envs=4096, env_spacing=4.0
    )
    # Basic settings
    observations: X7ObservationsCfg = X7ObservationsCfg()
    actions: X7ActionsAllCfg = X7ActionsAllCfg()
    events: X7LiftRandomEventCfg = X7LiftRandomEventCfg()
    # MDP settings
    rewards: _X7AliveRewardsCfg = _X7AliveRewardsCfg()
    terminations: _TimeoutTerminationsCfg = _TimeoutTerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 30
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
