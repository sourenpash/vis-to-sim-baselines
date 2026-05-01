# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp import (
    InitialStateRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


@configclass
class MYInitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = recorders.MYInitialStateRecorder


@configclass
class MYPostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.MYPostStepStatesRecorder


@configclass
class MYPreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = recorders.MYPreStepActionsRecorder


@configclass
class MYPreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = recorders.MYPreStepFlatPolicyObservationsRecorder


@configclass
class ImagePolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation image recorder term."""

    class_type: type[RecorderTerm] = recorders.ImagePolicyObservationsRecorder


@configclass
class DepthPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation depth recorder term."""

    class_type: type[RecorderTerm] = recorders.DepthPolicyObservationsRecorder


@configclass
class ImagePolicyJpgObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation image recorder term."""

    class_type: type[RecorderTerm] = recorders.ImagePolicyJpgObservationsRecorder


@configclass
class MyInitialCameraInfoRecorderCfg(RecorderTermCfg):
    class_type = recorders.MyInitialCameraInfoRecorder


# @configclass
# class ActionStateImageDepthRecorderManagerCfgJpg(RecorderManagerBaseCfg):
#     """Recorder configurations for recording actions and states."""

#     record_initial_state = MYInitialStateRecorderCfg()
#     record_post_step_states = MYPostStepStatesRecorderCfg()
#     record_pre_step_actions = MYPreStepActionsRecorderCfg()
#     record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
#     record_image_policy_observations = ImagePolicyJpgObservationsRecorderCfg()
#     record_depth_policy_observations = DepthPolicyObservationsRecorderCfg()


@configclass
class PointCloudObsRecorderCfg(RecorderTermCfg):
    class_type = recorders.PointCloudObsRecorder


@configclass
class WbcStepRecorderCfg(RecorderTermCfg):
    class_type = recorders.WbcStepRecorder


@configclass
class WbcTargetRecorderCfg(RecorderTermCfg):
    class_type = recorders.WbcTargetRecorder


@configclass
class ActionStateImageDepthRecorderManagerCfg(RecorderManagerBaseCfg):
    record_initial_state = MYInitialStateRecorderCfg()
    record_post_step_states = MYPostStepStatesRecorderCfg()
    record_pre_step_actions = MYPreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = (
        PreStepFlatPolicyObservationsRecorderCfg()
    )
    record_image_policy_observations = ImagePolicyObservationsRecorderCfg()
    record_depth_policy_observations = DepthPolicyObservationsRecorderCfg()
    record_camera_info = MyInitialCameraInfoRecorderCfg()
    record_pcds = PointCloudObsRecorderCfg()
    record_wbcstep = WbcStepRecorderCfg()
    record_wbctarget = WbcTargetRecorderCfg()


@configclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    record_initial_state = MYInitialStateRecorderCfg()
    record_post_step_states = MYPostStepStatesRecorderCfg()
    record_pre_step_actions = MYPreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = (
        PreStepFlatPolicyObservationsRecorderCfg()
    )
    record_wbcstep = WbcStepRecorderCfg()
    record_wbctarget = WbcTargetRecorderCfg()
