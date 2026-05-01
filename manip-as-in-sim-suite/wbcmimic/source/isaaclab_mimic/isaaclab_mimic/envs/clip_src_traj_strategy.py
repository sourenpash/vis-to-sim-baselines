# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.utils import configclass


class ClipSrcTrajCBase:
    def __init__(self, cfg: "ClipSrcTrajBaseCfg"):
        self.cfg = cfg

    def cilp_src_traj(
        self,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor,
        src_traj: torch.Tensor,
        src_gripper_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip the source trajectory.
        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_traj (torch.Tensor): Trajectory of the source demo in (N, 4, 4) poses
        Returns:
            clipped source trajectory (torch.Tensor): Trajectory of the clipped source demo in (N, 4, 4) poses
        """
        assert len(src_gripper_actions) == len(src_traj)
        return src_traj, src_gripper_actions


@configclass
class ClipSrcTrajBaseCfg:
    class_type = ClipSrcTrajCBase


class ClipSrcTrajLastStep(ClipSrcTrajCBase):
    def __init__(self, cfg: "LastStepClipCfg"):
        super().__init__(cfg)
        self.last_step_num = cfg.last_step_num

    def cilp_src_traj(
        self,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor,
        src_traj: torch.Tensor,
        src_gripper_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            src_traj[-self.last_step_num :],
            src_gripper_actions[-self.last_step_num :],
        )


@configclass
class LastStepClipCfg(ClipSrcTrajBaseCfg):
    class_type = ClipSrcTrajLastStep

    last_step_num: int = 30
    """Number of last steps to keep."""


class ClipFarPoses(ClipSrcTrajCBase):
    def __init__(self, cfg: "ClipFarPosesCfg"):
        super().__init__(cfg)
        self.cfg = cfg

    def clip_src_traj(
        self,
        eef_pose: torch.Tensor,
        object_pose: torch.Tensor,
        src_traj: torch.Tensor,
        src_gripper_actions: torch.Tensor,
    ):
        indice_to_keep = (
            torch.norm(
                src_traj[:, :3, 3] - object_pose[:3, 3].repeat(src_traj.shape[0], 1)
            )
            < self.cfg.distance_threshold
        )
        first_indice = torch.where(indice_to_keep)[0][0]
        return src_traj[first_indice:], src_gripper_actions[first_indice:]


@configclass
class ClipFarPosesCfg(ClipSrcTrajBaseCfg):
    distance_threshold = 0.2
    class_type = ClipFarPoses
