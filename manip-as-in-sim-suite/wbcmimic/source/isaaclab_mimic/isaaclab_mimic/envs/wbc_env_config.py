# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import MISSING
from typing import List

from isaaclab.envs import ManagerBasedRLEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .clip_src_traj_strategy import ClipSrcTrajBaseCfg


@configclass
class WbcSubTaskConfig(SubTaskConfig):
    clip_src_traj_strategy: ClipSrcTrajBaseCfg = ClipSrcTrajBaseCfg()
    wbc_max_step: int = 5
    """wbc max step at the middle step(neither start nor last)"""
    wbc_max_step_start: int = 100
    """wbc max step at the first step"""
    wbc_max_step_last: int = 100
    """wbc max step at the last step"""
    slice_to_last_success_step: bool = False
    """whether the end of subtask is the last success step or the first success step"""
    subtask_continous_step = -1
    """The minimum number of consecutive steps required to be considered as a subtask."""
    vel_multiplier: float = 1.0
    """Velocity multiplier for the WBC task, which is required for WBC tasks."""


@configclass
class RLWbcSubTaskConfig(SubTaskConfig):
    rl_env_config: ManagerBasedRLEnvCfg = MISSING
    """RL environment configuration, which is required for RL-based WBC tasks."""

    rl_max_step = 1000
    """Maximum number of steps for the RL environment."""

    rl_model_path: str = MISSING
    """Path to the RL model, which is required for RL-based WBC tasks."""

    clip_src_traj_strategy: ClipSrcTrajBaseCfg = ClipSrcTrajBaseCfg()
    """Strategy for clipping source trajectories, which is required for RL-based WBC tasks."""

    wbc_max_step = 100
    """wbc max step at the first step"""


@configclass
class KeyPointSubTaskConfig(SubTaskConfig):
    wbc_max_step = 100
    """wbc max step"""

    key_eef_list: List = MISSING
    """List of key end-effectors (x y z qw qx qy qz, in robot space), which is required for key point control."""

    key_gripper_action_list: List = MISSING
    """List of key end-effector actions, which is required for key point control."""

    def __post_init__(self):
        self.wbc_max_step_start = self.wbc_max_step
        self.wbc_max_step_last = self.wbc_max_step


class SubtaskControlMode(enum.Enum):
    """Enum for subtask control modes."""

    WBC = "wbc"
    RL = "rl"
    KEY_POINT = "key_point"

    @classmethod
    def is_rl_mode(cls, mode: "SubtaskControlMode") -> bool:
        """Check if the mode is RL-based."""
        return mode == cls.RL

    @classmethod
    def is_wbc_mode(cls, mode: "SubtaskControlMode") -> bool:
        """Check if the mode is WBC-based."""
        return mode == cls.WBC

    @classmethod
    def is_key_point_mode(cls, mode: "SubtaskControlMode") -> bool:
        """Check if the mode is key point based."""
        return mode == cls.KEY_POINT

    @classmethod
    def get_mode_by_subtask_config(
        cls, subtask_config: SubTaskConfig
    ) -> "SubtaskControlMode":
        """Get the control mode based on the subtask configuration."""
        if isinstance(subtask_config, RLWbcSubTaskConfig):
            return cls.RL
        elif isinstance(subtask_config, WbcSubTaskConfig):
            return cls.WBC
        elif isinstance(subtask_config, KeyPointSubTaskConfig):
            return cls.KEY_POINT
        else:
            raise ValueError(f"Unsupported subtask config type: {type(subtask_config)}")
