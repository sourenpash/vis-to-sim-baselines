# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to add mimic annotations to demos to be used as source demos for mimic dataset generation.
"""

# Launching Isaac Sim Simulator first.

import argparse
import inspect

import pinocchio as pin
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Annotate demonstrations for Isaac Lab environments."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-Mimic-Annotated-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--input_file",
    type=str,
    default="./datasets/put_bowl_in_microwaveandclose_record_514.hdf5",
    help="File name of the dataset to be annotated.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/put_bowl_in_microwaveandclose_record_514_replay_1.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument(
    "--auto",
    action="store_true",
    default=True,
    help="Automatically annotate subtasks.",
)
parser.add_argument(
    "--state_only",
    action="store_true",
    default=True,
    help="Whether only save state info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import gymnasium as gym
import isaaclab_mimic.envs  # noqa: F401
import torch

# Only enables inputs if this script is NOT headless mode
if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard

import isaaclab_mimic.tasks
import isaaclab_mimic.utils.tensor_utils as TensorUtils
import isaaclab_tasks  # noqa: F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import ManagerTermBase, RecorderTerm, RecorderTermCfg
from isaaclab.sim import RenderCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_mimic.envs import ManagerBasedRLMimicEnv
from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler
from isaaclab_mimic.utils.replay import compare_states, concatenate_state
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(
                self._env.action_manager.action
            ),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return (
            "obs/datagen_info/subtask_term_signals",
            self._env.get_subtask_term_signals(),
        )


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    global is_paused, current_action_index, subtask_indices

    if not args_cli.auto and len(args_cli.signals) == 0:
        if len(args_cli.signals) == 0:
            raise ValueError("Subtask signals should be provided for manual mode.")

    # Load input dataset to be annotated
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(
            f"The input dataset file {args_cli.input_file} does not exist."
        )
    if not args_cli.input_file.endswith(".hdf5"):
        dataset_file_handler = ZarrCompatibleDatasetFileHandler()
    else:
        dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    # get output directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)

    env_cfg.env_name = args_cli.task
    env_cfg.sim.render = RenderCfg(
        enable_translucency=False,
        enable_reflections=False,
        antialiasing_mode="FXAA",
        dlss_mode=0,
        enable_shadows=False,
        enable_ambient_occlusion=False,
        enable_dlssg=True,
    )
    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success

        env_cfg.terminations.success = None
    else:
        raise NotImplementedError(
            "No success termination term was found in the environment."
        )

    key_to_del = []
    for key in vars(env_cfg.observations).keys():
        if key not in ["policy", "subtask_terms"]:
            key_to_del.append(key)
    for key in key_to_del:
        delattr(env_cfg.observations, key)

    # Disable all termination terms
    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations
    env_cfg.recorders: MimicRecorderManagerCfg = MimicRecorderManagerCfg()
    if not args_cli.auto:
        # disable subtask term signals recorder term if in manual mode
        env_cfg.recorders.record_pre_step_subtask_term_signals = None
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if args_cli.state_only:
        if hasattr(env_cfg.observations, "images"):
            delattr(env_cfg.observations, "images")
        if hasattr(env_cfg.observations, "depths"):
            delattr(env_cfg.observations, "depths")
        if hasattr(env_cfg.observations, "policy_infer"):
            delattr(env_cfg.observations, "policy_infer")
        if hasattr(env_cfg.observations, "camera_info"):
            delattr(env_cfg.observations, "camera_info")
        item_to_delete = []
        # for item in vars(env_cfg.scene).keys():
        # if "camera" in item:
        #     item_to_delete.append(item)
        for item in item_to_delete:
            delattr(env_cfg.scene, item)
    # create environment from loaded config
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if inspect.isclass(success_term.func):
        success_term.func = success_term.func(env=env, cfg=env_cfg)
    if not isinstance(env.unwrapped, ManagerBasedRLMimicEnv):
        raise ValueError(
            "The environment should be derived from ManagerBasedRLMimicEnv"
        )

    if args_cli.auto:
        # check if the mimic API env.unwrapped.get_subtask_term_signals() is implemented
        if (
            env.unwrapped.get_subtask_term_signals.__func__
            is ManagerBasedRLMimicEnv.get_subtask_term_signals
        ):
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )

    # reset environment
    env.reset()
    if isinstance(success_term.func, ManagerTermBase):
        success_term.func.reset()

    # simulate environment -- run everything in inference mode
    exported_episode_count = 0
    processed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Iterate over the episodes in the loaded dataset file
            for episode_index, episode_name in enumerate(
                dataset_file_handler.get_episode_names()
            ):
                processed_episode_count += 1
                subtask_indices = []
                print(
                    f"\nAnnotating episode #{episode_index} ({episode_name}). Warm up the environment..."
                )
                episode = dataset_file_handler.load_episode(
                    episode_name, env.unwrapped.device
                )
                episode_data = episode.data
                episode_data["actions"] = episode_data["actions"][1:]

                # warm up the environment
                env.unwrapped.recorder_manager.reset()
                env.reset()
                for _ in range(50):
                    default_actions = episode_data["actions"][0:1]
                    env.step(default_actions)
                print(f"Start to annotate episode #{episode_index} ({episode_name})")

                # read initial state from the loaded episode
                initial_state = episode_data["initial_state"]
                env.unwrapped.recorder_manager.reset()
                success_term.func.reset()
                env.unwrapped.reset_to(initial_state, None, is_relative=True)

                # replay actions from this episode
                actions = episode_data["actions"]
                first_action = True
                for action_index, action in enumerate(actions):
                    current_action_index = action_index
                    if first_action:
                        first_action = False
                    action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
                    obs, reward, terminated, truncated, info = env.step(
                        torch.Tensor(action_tensor)
                    )
                    # for k, v in obs["subtask_terms"].items():
                    #     print(f"{k}: {v}")
                    success_term.func(env, **success_term.params)
                    current_runtime_state = env.scene.get_state(is_relative=True)
                    current_runtime_state = TensorUtils.map_tensor(
                        current_runtime_state, lambda x: x.squeeze(0)
                    )
                    state_from_dataset = episode.get_state(action_index)
                    states_matched = compare_states(
                        state_from_dataset,
                        current_runtime_state,
                        num_envs=1,
                        device=env.device,
                    )
                    states_matched = bool(states_matched)
                    # if not states_matched:
                    #     # print(
                    #     #     f"State mismatch at action index {action_index}. Resetting to the state from the dataset."
                    #     # )
                    #     state_from_dataset = TensorUtils.map_tensor(state_from_dataset, lambda x: x.unsqueeze(0))
                    #     env.scene.reset_to(state_from_dataset, torch.tensor([0]), is_relative=True)

                is_episode_annotated_successfully = False
                if not args_cli.auto:
                    print(f"\tSubtasks marked at action indices: {subtask_indices}")
                    if len(args_cli.signals) != len(subtask_indices):
                        raise ValueError(
                            f"Number of annotated subtask signals {len(subtask_indices)} should be equal               "
                            f"                          to number of subtasks {len(args_cli.signals)}"
                        )
                    annotated_episode = env.unwrapped.recorder_manager.get_episode(0)
                    for subtask_index in range(len(args_cli.signals)):
                        # subtask termination signal is false until subtask is complete, and true afterwards
                        subtask_signals = torch.ones(len(actions), dtype=torch.bool)
                        subtask_signals[: subtask_indices[subtask_index]] = False
                        annotated_episode.add(
                            f"obs/datagen_info/subtask_term_signals/{args_cli.signals[subtask_index]}",
                            subtask_signals,
                        )
                    is_episode_annotated_successfully = True
                else:
                    # use squential subtask success term signals to determine if the episode is annotated successfully
                    is_episode_annotated_successfully = True

                    # # check if all the subtask term signals are annotated
                    # annotated_episode = env.unwrapped.recorder_manager.get_episode(0)
                    # subtask_term_signal_dict = annotated_episode.data["obs"][
                    #     "datagen_info"
                    # ]["subtask_term_signals"]
                    # for signal_name, signal_flags in subtask_term_signal_dict.items():
                    #     if not torch.any(signal_flags):
                    #         is_episode_annotated_successfully = False
                    #         print(
                    #             f'\tDid not detect completion for the subtask "{signal_name}".'
                    #         )

                if not bool(success_term.func(env, **success_term.params)[0]):
                    is_episode_annotated_successfully = False
                    print("\tThe final task was not completed.")

                if is_episode_annotated_successfully:
                    # set success to the recorded episode data and export to file
                    env.unwrapped.recorder_manager.set_success_to_episodes(
                        None,
                        torch.tensor(
                            [[True]], dtype=torch.bool, device=env.unwrapped.device
                        ),
                    )
                    env.unwrapped.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    print("\tExported the annotated episode.")
                else:
                    print(
                        "\tSkipped exporting the episode due to incomplete subtask annotations."
                    )
            break

    print(
        f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
        f" episode{'s' if exported_episode_count > 1 else ''}."
    )
    print("Exiting the app.")

    # Close environment after annotation is complete
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
