# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab_mimic.datagen.data_generator import DataGenerator

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.managers.mimic_event_manager import MimicEventManager
from isaaclab_mimic.utils.datasets import DistDatasetFileHandlerSlave


class ManagerBasedRLMimicEnv(ManagerBasedRLEnv):
    """The superclass for the Isaac Lab Mimic environments.

    This class inherits from :class:`ManagerBasedRLEnv` and provides a template for the functions that
    need to be defined to run the Isaac Lab Mimic data generation workflow. The Isaac Lab data generation
    pipeline, inspired by the MimicGen system, enables the generation of new datasets based on a few human
    collected demonstrations. MimicGen is a novel approach designed to automatically synthesize large-scale,
    rich datasets from a sparse set of human demonstrations by adapting them to new contexts. It manages to
    replicate the benefits of large datasets while reducing the immense time and effort usually required to
    gather extensive human demonstrations.

    The MimicGen system works by parsing demonstrations into object-centric segments. It then adapts
    these segments to new scenes by transforming each segment according to the new scene’s context, stitching
    them into a coherent trajectory for a robotic end-effector to execute. This approach allows learners to train
    proficient agents through imitation learning on diverse configurations of scenes, object instances, etc.

    Key Features:
        - Efficient Dataset Generation: Utilizes a small set of human demos to produce large scale demonstrations.
        - Broad Applicability: Capable of supporting tasks that require a range of manipulation skills, such as
          pick-and-place and interacting with articulated objects.
        - Dataset Versatility: The synthetic data retains a quality that compares favorably with additional human demos.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._subtask_term_signal_to_idx = {}
        self._subtasks = {}
        self.wbc_target_buf: Dict[str, torch.tensor] = {}
        for eef_name, subtask_cfgs in self.cfg.subtask_configs.items():
            self._subtasks[eef_name] = []
            self.wbc_target_buf[eef_name] = torch.zeros(
                (self.num_envs, 8), device=self.device
            )
            self._subtask_term_signal_to_idx[eef_name] = {}
            for index, subtask_cfg in enumerate(subtask_cfgs):
                subtask_cfg: SubTaskConfig
                self._subtasks[eef_name].append(subtask_cfg.subtask_term_signal)
                self._subtask_term_signal_to_idx[eef_name][
                    subtask_cfg.subtask_term_signal
                ] = index
        self._data_generator = None
        self._enable_mimic_event_manager = False
        if hasattr(self.cfg, "mimic_events"):
            self.mimic_event_manager = MimicEventManager(self.cfg.mimic_events, self)
            self._enable_mimic_event_manager = True
        self.wbc_step_buf = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

    def step(self, action):
        if self._enable_mimic_event_manager:
            # Call the mimic event manager to handle events
            self.mimic_event_manager.pre_step()
        res = super().step(action)
        if self._enable_mimic_event_manager:
            # Call the mimic event manager to handle events
            self.mimic_event_manager.post_step()
        return res

    def reset(self, seed=None, env_ids=None, options=None):
        if self._enable_mimic_event_manager:
            # Call the mimic event manager to handle events
            self.mimic_event_manager.pre_reset()
        res = super().reset(seed, env_ids, options)
        if self._enable_mimic_event_manager:
            # Call the mimic event manager to handle events
            self.mimic_event_manager.post_reset()
        return res

    @property
    def subtask_term_signal_to_idx(self):
        return self._subtask_term_signal_to_idx

    @property
    def subtasks(self):
        return self._subtasks

    @property
    def data_generator(self):
        """
        The data generator used to generate data for the environment. This is used to generate data for the environment.
        """
        return self._data_generator

    def register_data_generator(
        self,
        data_generator: "DataGenerator",
    ):
        """
        Register a data generator to the environment. This is used to generate data for the environment.

        Args:
            data_generator: The data generator to register.
        """
        self._data_generator = data_generator

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        raise NotImplementedError

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        vel_multiplier: float = 1.0,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to compute the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        raise NotImplementedError

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        raise NotImplementedError

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        raise NotImplementedError

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        rigid_object_states = self.scene.get_state(is_relative=True)["rigid_object"]
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3],
                PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
            )
        return object_pose_matrix

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        raise NotImplementedError

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.spec.id, type=2, env_kwargs=dict())

    def set_data_queue(self, episode_data_queue, episode_data_queue_failed=None):
        assert isinstance(
            self.recorder_manager._dataset_file_handler, DistDatasetFileHandlerSlave
        ), "Data queue only works for DistDatasetFileHandlerSlave, but got {}".format(
            type(self.recorder_manager._dataset_file_handler)
        )
        self.recorder_manager._dataset_file_handler.set_data_queue(episode_data_queue)
        if (
            self.cfg.recorders.dataset_export_mode
            == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        ):
            self.recorder_manager._failed_episode_dataset_file_handler.set_data_queue(
                episode_data_queue_failed
            )
