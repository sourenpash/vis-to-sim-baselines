# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import cv2
import isaaclab_mimic.utils.tensor_utils as TensorUtils
import numpy as np
import torch
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab_mimic.utils.pcd_utils import create_pointcloud_from_rgbd_batch


class ImagePolicyJpgObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def _tensor_to_jpg_num(self, tensor: torch.Tensor):
        images_jpg = []
        for i, image in enumerate(tensor):
            image_np = image.cpu().numpy().astype(np.uint8)
            image_jpg = cv2.imencode(".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, 20])[
                1
            ]
            images_jpg.append(image_jpg)
        images_jpg = torch.from_numpy(np.array(images_jpg))
        return images_jpg

    def record_pre_step(self):
        images = self._env.obs_buf["images"]
        images = TensorUtils.map_tensor(images, self._tensor_to_jpg_num)
        return "obs/images", images


class ImagePolicyObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        if "images" not in self._env.obs_buf:
            return None, None
        images = self._env.obs_buf["images"]
        images = TensorUtils.to_device(images, "cpu")
        return "obs/images", images


class DepthPolicyObservationsRecorder(RecorderTerm):

    def record_pre_step(self):
        if "depths" not in self._env.obs_buf:
            return None, None
        depths = self._env.obs_buf["depths"]
        depths = TensorUtils.to_device(depths, "cpu")
        return "obs/depths", depths


class PointCloudObsRecorder(RecorderTerm):

    def record_pre_step(self):
        if (
            "policy_infer" not in self._env.obs_buf
            or "pointcloud" not in self._env.obs_buf["policy_infer"]
        ):
            return None, None
        pointcloud = self._env.obs_buf["policy_infer"]["pointcloud"]
        pos = pointcloud[..., 0]
        color = pointcloud[..., 1]
        res = {"pos": pos, "color": color}
        res = TensorUtils.to_device(res, "cpu")
        return "obs/pointcloud", res


class MYInitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", TensorUtils.to_device(
            extract_env_ids_values(self._env.scene.get_state(is_relative=True)), "cpu"
        )


class MyInitialCameraInfoRecorder(RecorderTerm):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._recorded = False

    def record_post_reset(self, envs_id):
        self._recorded = False
        return None, None

    def record_post_step(self):
        if not self._recorded:
            if "camera_info" not in self._env.obs_buf:
                return None, None
            camera_info = self._env.obs_buf["camera_info"]
            # camera_extrs = self._env.obs_buf["camera_extr"]
            camera_info = TensorUtils.to_device(camera_info, "cpu")
            camera_names = set()
            for key in camera_info:
                key: str
                camera_name = key[:-5]
                if camera_name not in camera_names:
                    camera_names.add(camera_name)
            result = {camera_name: {} for camera_name in camera_names}
            for camera_name in camera_names:
                result[camera_name]["intrinsic"] = camera_info[f"{camera_name}_intr"]
                result[camera_name]["extrinsic"] = camera_info[f"{camera_name}_extr"]
            self._recorded = True
            return "camera_info", result
        else:
            return None, None


class MYPostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", TensorUtils.to_device(
            self._env.scene.get_state(is_relative=True), device="cpu"
        )


class MYPreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        action = self._env.action_manager.action
        # temporarily hard code for gripper action
        action[:, -1] = action[:, -1] > 0.035
        assert (
            not torch.isnan(action).any().item()
        ), "Actions contain NaN values, please check your action space and action generation logic."
        return "actions", TensorUtils.to_device(action, "cpu")


class MYPreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", TensorUtils.to_device(self._env.obs_buf["policy"], "cpu")


class WbcTargetRecorder(RecorderTerm):
    def record_post_step(self):
        if not hasattr(self._env, "wbc_target_buf"):
            return None, None
        return "wbc_target", TensorUtils.to_device(self._env.wbc_target_buf, "cpu")


class WbcStepRecorder(RecorderTerm):
    """Recorder term that records the WBC step."""

    def record_post_step(self):
        wbc_step = getattr(
            self._env,
            "wbc_step_buf",
            torch.ones(
                (self._env.num_envs,), dtype=torch.bool, device=self._env.device
            ),
        )
        wbc_step = TensorUtils.to_device(wbc_step, "cpu")
        return "wbc_step", wbc_step
