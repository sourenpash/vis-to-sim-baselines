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
import os
import time
from pathlib import Path

import gymnasium as gym
import pinocchio as pin
import torch
import torch.multiprocessing as mp
import tqdm
from isaaclab.app import AppLauncher
from torch.multiprocessing import JoinableQueue

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Annotate demonstrations for Isaac Lab environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to replay episodes."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="File name of the dataset to be annotated.",
)
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    default="./datasets/dataset.hdf5",
    help="File name of the dataset to be annotated.",
)
parser.add_argument(
    "--signals",
    type=str,
    nargs="+",
    default=[],
    help="Sequence of subtask termination signals for all except last subtask",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--n_procs",
    type=int,
    default=8,
    help="Number of procs.",
)
parser.add_argument(
    "--episode_count", type=int, default=1000, help="Number of episodes"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def compare_states(state_from_dataset, runtime_state, device, num_envs):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = torch.tensor([True] * num_envs, device=device)
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][
                    state_name
                ]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(
                        f"State shape of {state_name} for asset {asset_name} don't match"
                    )
                states_matched = torch.logical_and(
                    states_matched,
                    (
                        abs(dataset_asset_state.to(device) - runtime_asset_state) < 0.01
                    ).all(dim=1),
                )
    return states_matched


def concatenate_state(state_lst):
    res = {}
    dummy_state = state_lst[0]
    env_nums = len(state_lst)
    for asset_type in ["articulation", "rigid_object"]:
        res[asset_type] = {}
        for asset_name in dummy_state[asset_type].keys():
            res[asset_type][asset_name] = {}
            for state_name in dummy_state[asset_type][asset_name].keys():
                if state_name not in res[asset_type][asset_name]:
                    res[asset_type][asset_name][state_name] = torch.empty(
                        (
                            env_nums,
                            *dummy_state[asset_type][asset_name][state_name].shape,
                        )
                    )
                for i in range(env_nums):
                    res[asset_type][asset_name][state_name][i] = state_lst[i][
                        asset_type
                    ][asset_name][state_name]
    return res


def save_episode_data(
    rank,
    episode_data_queue: JoinableQueue,
    total_episodes,
    dataset_export_dir_path,
    dataset_filename,
    env_name,
    stop_queue: mp.Queue,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)

    print(">>>Save episode data process rank:", rank)

    app_launcher = AppLauncher(args_cli)
    from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler

    dataset_file_handler = ZarrCompatibleDatasetFileHandler()
    dataset_file_handler.create(
        os.path.join(dataset_export_dir_path, dataset_filename), env_name=env_name
    )

    idx = 0
    while idx < total_episodes:
        # print("getting new data...", flush=True)
        episode_data = episode_data_queue.get()
        # print("getting one data", flush=True)
        dataset_file_handler.write_episode(episode_data)
        dataset_file_handler.flush()
        idx += 1
        episode_data_queue.task_done()
        print("Finish write the {} episode!".format(idx), flush=True)

    dataset_file_handler.close()
    for i in range(args_cli.n_procs):
        stop_queue.put(1)
    time.sleep(10)


def main(
    rank,
    world_size,
    episode_data_queue: JoinableQueue,
    episode_indices_to_replay_lst,
    stop_queue: mp.Queue,
):
    try:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        print(">>>Main process rank:", rank)

        # launch the simulator
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        """Rest everything follows."""

        import isaaclab_mimic.envs  # noqa: F401
        import isaaclab_mimic.tasks
        import isaaclab_mimic.utils.tensor_utils as TensorUtils
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab.managers import DatasetExportMode
        from isaaclab.utils.datasets import EpisodeData
        from isaaclab_mimic.envs.mdp import ActionStateImageDepthRecorderManagerCfg
        from isaaclab_mimic.utils.datasets import (
            DistDatasetFileHandlerSlave,
            ZarrCompatibleDatasetFileHandler,
        )
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        """Add Isaac Lab Mimic annotations to the given demo dataset file."""
        # Load input dataset to be annotated
        if not os.path.exists(args_cli.input_file):
            raise FileNotFoundError(
                f"The input dataset file {args_cli.input_file} does not exist."
            )
        dataset_file_handler = ZarrCompatibleDatasetFileHandler()
        dataset_file_handler.open(args_cli.input_file)
        env_name = dataset_file_handler.get_env_name()
        episode_count = dataset_file_handler.get_num_episodes()

        if episode_count == 0:
            print("No episodes found in the dataset.")
            exit()

        if args_cli.task is not None:
            env_name = args_cli.task
        if env_name is None:
            raise ValueError(
                "Task/env name was not specified nor found in the dataset."
            )

        num_envs = args_cli.num_envs
        if args_cli.n_procs > 1:
            args_cli.device = f"cuda:{app_launcher.local_rank}"
            args_cli.device_name = f"cuda:{app_launcher.local_rank}"
            args_cli.multi_gpu = True
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
            env_name, device=args_cli.device, num_envs=num_envs
        )

        env_cfg.env_name = args_cli.task

        # extract success checking function to invoke manually
        success_term = None
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            raise NotImplementedError(
                "No success termination term was found in the environment."
            )

        # Disable all termination terms
        env_cfg.terminations = None

        # setup recorder
        output_file = Path(args_cli.output_file)
        dataset_file_handler_output = DistDatasetFileHandlerSlave
        env_cfg.recorders = ActionStateImageDepthRecorderManagerCfg()
        env_cfg.recorders.dataset_file_handler_class_type = dataset_file_handler_output
        os.makedirs(output_file.parent, exist_ok=True)
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        env_cfg.recorders.dataset_export_dir_path = str(output_file.parent)
        env_cfg.recorders.dataset_filename = output_file.name
        # create environment from loaded config
        print("before make env")
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print("after make env")
        env.set_data_queue(episode_data_queue)
        # reset environment
        env.reset()

        # simulate environment -- run everything in inference mode
        exported_episode_count = 0
        processed_episode_count = 0
        replayed_episode_count = 0
        idle_action = env.cfg.mimic_config.default_actions.repeat(num_envs, 1)
        episode_names = list(dataset_file_handler.get_episode_names())
        episode_indices_to_replay = [i for i in episode_indices_to_replay_lst[rank]]
        tqdm_bar = tqdm.tqdm(range(len(episode_indices_to_replay)))
        start_time = time.time()
        with torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                # Iterate over the episodes in the loaded dataset file
                env_episode_data_map = {
                    index: EpisodeData() for index in range(num_envs)
                }
                has_next_action = True
                while has_next_action:
                    actions = idle_action
                    has_next_action = False
                    for env_id in range(num_envs):
                        env_next_action = env_episode_data_map[env_id].get_next_action()
                        if env_next_action is None:
                            next_episode_index = None
                            while episode_indices_to_replay:
                                next_episode_index = episode_indices_to_replay.pop(0)
                                tqdm_bar.update(1)
                                if next_episode_index < episode_count:
                                    break
                                next_episode_index = None

                            if next_episode_index is not None:
                                replayed_episode_count += 1
                                print(
                                    f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}"
                                )
                                episode_data = dataset_file_handler.load_episode(
                                    episode_names[next_episode_index], env.device
                                )
                                env_episode_data_map[env_id] = episode_data
                                # Set initial state for the new episode
                                initial_state = episode_data.get_initial_state()
                                env.reset_to(
                                    initial_state,
                                    torch.tensor([env_id], device=env.device),
                                    is_relative=True,
                                )
                                # Get the first action for the new episode
                                env_next_action = env_episode_data_map[
                                    env_id
                                ].get_next_action()
                                has_next_action = True
                            else:
                                continue
                        else:
                            has_next_action = True
                        actions[env_id] = env_next_action
                    env.step(actions)
                    state_from_dataset = [
                        env_episode_data_map[env_id].get_state(
                            env_episode_data_map[env_id].next_action_index - 1
                        )
                        for env_id in range(num_envs)
                    ]
                    state_from_dataset = concatenate_state(state_from_dataset)
                    state_from_dataset = TensorUtils.to_device(
                        state_from_dataset, env.device
                    )
                    if state_from_dataset is not None:
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched = compare_states(
                            state_from_dataset,
                            current_runtime_state,
                            num_envs=num_envs,
                            device=env.device,
                        )
                        if not states_matched.all():
                            # print("\t- mismatched.")
                            # print(comparison_log)
                            env_id_tensor = torch.where(~states_matched)[0].to(
                                env.device
                            )
                            state_from_dataset_to_reset = TensorUtils.map_tensor(
                                state_from_dataset, lambda x: x[env_id_tensor]
                            )
                            env.scene.reset_to(
                                state_from_dataset_to_reset,
                                env_id_tensor,
                                is_relative=True,
                            )

                    if bool(success_term.func(env, **success_term.params).any()):
                        env_id_tensor = torch.where(
                            success_term.func(env, **success_term.params)
                        )[0]
                        env.recorder_manager.set_success_to_episodes(
                            env_id_tensor,
                            torch.ones_like(env_id_tensor).to(
                                dtype=bool, device=env.device
                            ),
                        )
                        env.recorder_manager.export_episodes(env_id_tensor)
                        print("Episode completed successfully.")
                        print(f"Time used now: {time.time() - start_time}")

                break

        print(
            f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
            f" episode{'s' if exported_episode_count > 1 else ''}."
        )
        print("Exiting the app.")
        stop_queue.get()
        shared_queue.join()
        # Close environment after annotation is complete
        env.close()
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29523"
    mp.set_start_method("spawn")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    from isaaclab_mimic.datagen.utils import (
        get_env_name_from_dataset,
        setup_output_paths,
    )

    # 创建共享队列
    shared_queue = JoinableQueue()
    last_queue = mp.Queue()
    world_size = args_cli.n_procs
    episode_indices_to_replay_lst = [[] for i in range(world_size)]

    episode_count = args_cli.episode_count
    for i in range(episode_count):
        episode_indices_to_replay_lst[i % world_size].append(i)

    print("begin spawn main")
    mp.spawn(
        main,
        args=(world_size, shared_queue, episode_indices_to_replay_lst, last_queue),
        nprocs=world_size,
        join=False,
        daemon=True,
    )

    output_file = Path(args_cli.output_file)
    dataset_export_dir_path = str(output_file.parent)
    dataset_filename = output_file.name
    env_name = args_cli.task
    try:
        mp.spawn(
            save_episode_data,
            args=(
                shared_queue,
                episode_count,
                dataset_export_dir_path,
                dataset_filename,
                env_name,
                last_queue,
            ),
            join=True,
            daemon=True,
        )
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
