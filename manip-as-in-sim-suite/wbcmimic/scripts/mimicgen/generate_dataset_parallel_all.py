# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

# Launching Isaac Sim Simulator first.

import argparse
import asyncio
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
import pinocchio as pin
import torch
import torch.multiprocessing as mp
from isaaclab.app import AppLauncher
from loguru import logger as lgr
from torch.multiprocessing import Event, JoinableQueue

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Generate demonstrations for Isaac Lab environments."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--generation_num_trials",
    type=int,
    help="Number of demos to be generated.",
    default=None,
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to instantiate for generating datasets.",
)
parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    required=True,
    help="File path to the source dataset file.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
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
    "--state_only",
    action="store_true",
    help="Whether only save state info.",
)
parser.add_argument(
    "--record_all",
    action="store_true",
    help="Whether to record all data.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
)
parser.add_argument(
    "--mimic_mode",
    type=str,
    choices=["origin", "wbc"],
    default="wbc",
    help="Choose control mode: origin, wbc (default: origin) in mimicgen",
)
parser.add_argument(
    "--failed_save_prob",
    type=float,
    default=0.1,
    help="Probability of saving failed episodes when record_all is True. Default is 0.1. Set to 0.0 to disable saving failed episodes.",
)
parser.add_argument(
    "--batch_splits",
    type=int,
    nargs="+",
    default=None,
    help="List of batch sizes for splitting successful episodes into multiple zarr files. E.g., [100, 200, 400, 100] will create 4 files with .001.zarr, .002.zarr, .003.zarr, .004.zarr suffixes. The last number can be -1 to include all remaining episodes. If not provided, equivalent to [-1] (single file).",
)
parser.add_argument(
    "--extra_wbc_params_path",
    type=str,
    default=None,
    help="Path to extra WBC parameters file. Default is None, which means no extra parameters are used.",
)
parser.add_argument(
    "--no_save_images",
    default=False,
    action="store_true",
    help="Whether to save images in the dataset. Default is False, which means images will be saved. Set to True to disable saving images.",
)

parser.add_argument(
    "--camera_names",
    type=str,
    nargs="+",
    default=["camera_0"],
    help="List of camera names to save images from. Default is ['camera_0']. If no cameras are specified, images will not be saved.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app


def save_episode_data(
    rank,
    episode_data_queue: JoinableQueue,
    total_episodes,
    dataset_export_dir_path,
    dataset_filename,
    env_name,
    all_done_event: Event,
    save_success_process=True,
    prob=1.0,
    batch_splits=None,
):
    try:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)

        lgr.trace(f">>>Save episode data process rank: {rank}")

        app_launcher = AppLauncher(args_cli)
        from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler

        # Handle batch_splits parameter
        if batch_splits is None:
            batch_splits = [total_episodes]  # Default: single file with all episodes

        # batch_splits should already be validated at this point
        if sum(batch_splits) != total_episodes:
            raise ValueError(
                f"Sum of batch_splits ({sum(batch_splits)}) must equal total_episodes ({total_episodes})"
            )

        # Create dataset file handlers for each batch
        dataset_file_handlers = []
        batch_episode_counts = []

        for batch_idx, batch_size in enumerate(batch_splits):
            if batch_size <= 0:
                continue

            # Generate batch filename
            base_name, ext = os.path.splitext(dataset_filename)
            if len(batch_splits) > 1:
                batch_filename = f"{base_name}.{batch_idx+1:03d}{ext}"
            else:
                batch_filename = dataset_filename

            # Create handler for this batch
            handler = ZarrCompatibleDatasetFileHandler()
            handler.create(
                os.path.join(dataset_export_dir_path, batch_filename), env_name=env_name
            )
            dataset_file_handlers.append(handler)
            batch_episode_counts.append(0)

            lgr.info(
                f"Created batch {batch_idx+1} handler for {batch_size} episodes: {batch_filename}"
            )

        current_batch_idx = 0
        total_saved = 0
        completed_batches = []  # Track completed batches

        while total_saved < total_episodes:
            episode_data = episode_data_queue.get()

            if np.random.rand() <= prob:
                lgr.trace("getting one data")

                # Find the appropriate batch handler
                while (
                    current_batch_idx < len(batch_splits)
                    and batch_episode_counts[current_batch_idx]
                    >= batch_splits[current_batch_idx]
                ):
                    current_batch_idx += 1

                if current_batch_idx >= len(dataset_file_handlers):
                    lgr.warning(f"No more batch handlers available. Skipping episode.")
                    episode_data_queue.task_done()
                    continue

                # Write to current batch handler
                dataset_file_handlers[current_batch_idx].write_episode(episode_data)
                dataset_file_handlers[current_batch_idx].flush()
                batch_episode_counts[current_batch_idx] += 1
                total_saved += 1

                episode_data_queue.task_done()
                lgr.info(
                    f"Finish write episode {total_saved} to batch {current_batch_idx+1} (episode {batch_episode_counts[current_batch_idx]} of {batch_splits[current_batch_idx]})!"
                )

                # Check if current batch is completed
                if (
                    batch_episode_counts[current_batch_idx]
                    >= batch_splits[current_batch_idx]
                ):
                    # Close the handler when batch is complete
                    dataset_file_handlers[current_batch_idx].close()
                    completed_batches.append(current_batch_idx)

                    # Create a .done file to signal completion
                    base_name, ext = os.path.splitext(dataset_filename)
                    if len(batch_splits) > 1:
                        batch_filename = f"{base_name}.{current_batch_idx+1:03d}{ext}"
                    else:
                        batch_filename = dataset_filename
                    batch_file_path = os.path.join(
                        dataset_export_dir_path, batch_filename
                    )
                    done_file_path = batch_file_path + ".done"

                    # Create the done file
                    with open(done_file_path, "w") as f:
                        f.write(
                            f"Batch {current_batch_idx+1} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        )

                    lgr.info(
                        f"Batch {current_batch_idx+1} completed and file closed. Created {done_file_path}"
                    )

            else:
                lgr.trace("skipping one data")
                episode_data_queue.task_done()
                lgr.trace("Finish skip episode!")

        # Close all remaining handlers
        for i, handler in enumerate(dataset_file_handlers):
            if i not in completed_batches:
                handler.close()

        lgr.info(f"Save process rank {rank} completed, waiting for all processes...")
        # Wait for all processes to complete before exiting
        if save_success_process:
            all_done_event.set()
        else:
            all_done_event.wait()
        lgr.info(f"Save process rank {rank} exiting...")
        # Do not close app_launcher.app to avoid affecting other processes
    except Exception as e:
        import traceback

        traceback.print_exc()
        lgr.error(f"Error in save process rank {rank}: {e}")


def parser_wbc_params(extra_wbc_params_path, env_cfg):
    if extra_wbc_params_path is None:
        return env_cfg
    if hasattr(env_cfg, "mimic_config") and hasattr(
        env_cfg.mimic_config, "wbc_solver_cfg"
    ):
        with open(extra_wbc_params_path, "rt") as f:
            config = json.load(f)
            env_cfg.mimic_config.wbc_solver_cfg.update(**config)
    else:
        lgr.warning(
            "Extra WBC parameters path provided, but environment config does not have a mimic_config with wbc_solver_cfg. "
            "WBC parameters will not be parsed."
        )


def main(
    rank,
    world_size,
    episode_data_queue: JoinableQueue,
    episode_data_queue_failed,
    all_done_event: Event,
):
    try:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        lgr.trace(">>>Main process rank:", rank)

        # launch the simulator
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        """Rest everything follows."""

        import isaaclab_mimic.envs  # noqa: F401
        import isaaclab_mimic.tasks
        import isaaclab_tasks  # noqa: F401
        from isaaclab_mimic.datagen.generation import (
            env_loop,
            setup_async_generation,
            setup_env_config,
        )
        from isaaclab_mimic.datagen.utils import (
            get_env_name_from_dataset,
            setup_output_paths,
        )
        from isaaclab_mimic.utils.datasets import DistDatasetFileHandlerSlave
        from isaaclab_mimic.utils.mp_wbc import setup_wbc_mp_env

        # =====================================================================================
        num_envs = args_cli.num_envs

        # Setup output paths and get env name
        output_dir, output_file_name, _, ext = setup_output_paths(args_cli.output_file)
        env_name = args_cli.task or get_env_name_from_dataset(args_cli.input_file)
        # multi-gpu training config
        if args_cli.n_procs > 1:
            args_cli.device = f"cuda:{app_launcher.local_rank}"
            args_cli.device_name = f"cuda:{app_launcher.local_rank}"
            args_cli.multi_gpu = True

        # Configure environment
        env_cfg, success_term, termination_term = setup_env_config(
            env_name=env_name,
            output_dir=output_dir,
            output_file_name=output_file_name,
            num_envs=num_envs,
            device=args_cli.device,
            generation_num_trials=args_cli.generation_num_trials,
            dataset_file_handler_class_type=DistDatasetFileHandlerSlave,
            ext=ext,
            state_only=args_cli.state_only,
            remove_terminations=False,
            record_all=args_cli.record_all,
        )
        env_cfg.seed = args_cli.seed
        if args_cli.n_procs > 1:
            # update env config device
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            env_cfg.seed += app_launcher.global_rank
            lgr.info("Seed:", env_cfg.seed)
        if args_cli.no_save_images:
            env_cfg.observations.images = None
        else:
            key_to_delete = []
            for camera_name in vars(env_cfg.observations.images):
                if "camera" in camera_name and camera_name not in args_cli.camera_names:
                    key_to_delete.append(camera_name)
            for camera_name in key_to_delete:
                delattr(env_cfg.observations.images, camera_name)
        key_to_delete = []
        for camera_name in vars(env_cfg.observations.depths):
            if "camera" in camera_name and camera_name not in args_cli.camera_names:
                key_to_delete.append(camera_name)
        for camera_name in key_to_delete:
            delattr(env_cfg.observations.depths, camera_name)

        # if int(os.environ.get("RANK", 0)) == 0: # only master process open a subprocess for saving data
        #     total_episodes = env_cfg.datagen_config.generation_num_trials * int(world_size)
        #     p = Process(
        #         target=save_episode_data, args=(int(world_size)+1, episode_data_queue, total_episodes, env_cfg.recorders.dataset_export_dir_path, env_cfg.recorders.dataset_filename, env_cfg.env_name)
        #     )
        #     p.daemon = True
        #     p.start()

        # create environment
        env_cfg.args_cli = args_cli
        env = gym.make(env_name, cfg=env_cfg)
        env.unwrapped.set_data_queue(episode_data_queue, episode_data_queue_failed)
        if not hasattr(env.unwrapped, "setup_mp") and args_cli.mimic_mode != "origin":
            lgr.critical("Error: env must have setup_mp method.")
            assert False
        lgr.trace("before setup env")
        if args_cli.mimic_mode != "origin":
            setup_wbc_mp_env(env.unwrapped)
        lgr.trace("after setup env")

        # set seed for generation
        random.seed(env.unwrapped.cfg.seed)
        np.random.seed(env.unwrapped.cfg.seed)
        torch.manual_seed(env.unwrapped.cfg.seed)

        # reset before starting
        env.reset()

        async_components = setup_async_generation(
            env=env.unwrapped,
            num_envs=args_cli.num_envs,
            input_file=args_cli.input_file,
            success_term=success_term,
            termination_term=termination_term,
            pause_subtask=args_cli.pause_subtask,
            use_mp=True if not args_cli.mimic_mode == "origin" else False,
            mimic_mode=args_cli.mimic_mode,
        )

        try:
            asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
            lgr.trace("Begin collecting")
            env_loop(
                env.unwrapped,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
        except asyncio.CancelledError:
            lgr.trace("Tasks were cancelled.")

        lgr.info(f"Data generation task at rank {rank} is done")
        episode_data_queue.join()
        if episode_data_queue_failed is not None:
            episode_data_queue_failed.join()
        lgr.info(
            f"Data generation process at rank {rank} completed, waiting for all processes..."
        )
        # Wait for all processes to complete before exiting
        all_done_event.wait()
        lgr.info(f"Data generation process at rank {rank} exiting...")
        # Do not close simulation_app to avoid affecting other processes
        # simulation_app.close()
    except Exception as e:
        lgr.error(str(e))
        import traceback

        traceback.print_exc()


def validate_batch_splits(batch_splits, total_episodes):
    """Validate and process batch_splits parameter."""
    if batch_splits is None:
        return [total_episodes]  # Single file with all episodes

    # Make a copy to avoid modifying the original
    batch_splits = batch_splits.copy()

    # Check for negative values (except -1 at the end)
    for i, split in enumerate(batch_splits):
        if split < 0 and not (i == len(batch_splits) - 1 and split == -1):
            raise ValueError(
                f"Only the last element of batch_splits can be -1, got {split} at position {i}"
            )

    # Process -1 at the end
    if batch_splits[-1] == -1:
        remaining_episodes = total_episodes - sum(batch_splits[:-1])
        if remaining_episodes < 0:
            raise ValueError(
                f"Sum of batch_splits excluding -1 ({sum(batch_splits[:-1])}) exceeds total_episodes ({total_episodes})"
            )
        batch_splits = batch_splits[:-1] + [remaining_episodes]

    # Check final sum
    if sum(batch_splits) != total_episodes:
        raise ValueError(
            f"Sum of batch_splits ({sum(batch_splits)}) must equal total_episodes ({total_episodes})"
        )

    # Check for zero or negative sizes
    for i, split in enumerate(batch_splits):
        if split <= 0:
            raise ValueError(
                f"Batch size at position {i} must be positive, got {split}"
            )

    return batch_splits


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(args_cli.n_procs)
    mp.set_start_method("spawn")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    from isaaclab_mimic.datagen.utils import (
        get_env_name_from_dataset,
        setup_output_paths,
    )

    # 创建共享队列和同步事件
    shared_queue = JoinableQueue(maxsize=4)
    shared_queue_failed = JoinableQueue(maxsize=4) if args_cli.record_all else None
    all_done_event = Event()

    world_size = args_cli.n_procs

    # Start data generation processes
    data_gen_procs = []
    ctx = mp.spawn(
        main,
        args=(world_size, shared_queue, shared_queue_failed, all_done_event),
        nprocs=world_size,
        join=False,
        daemon=False,
    )
    data_gen_procs.extend(ctx.processes)

    total_episodes = args_cli.generation_num_trials * int(world_size)

    dataset_export_dir_path, dataset_filename, output_failed_file_name, ext = (
        setup_output_paths(args_cli.output_file)
    )
    env_name = args_cli.task or get_env_name_from_dataset(args_cli.input_file)

    # Validate batch_splits parameter
    validated_batch_splits = validate_batch_splits(
        args_cli.batch_splits, total_episodes
    )

    # Start save processes
    save_procs = []
    try:
        if args_cli.record_all:
            # create a process to save failed data (no batch splitting for failed episodes)
            ctx_failed = mp.spawn(
                save_episode_data,
                args=(
                    shared_queue_failed,
                    total_episodes,
                    dataset_export_dir_path,
                    output_failed_file_name,
                    env_name,
                    all_done_event,
                    False,
                    args_cli.failed_save_prob,
                    None,
                ),
                nprocs=1,
                join=False,
                daemon=False,
            )
            save_procs.extend(ctx_failed.processes)
        # create a process to save successful data (with batch splitting)
        ctx_success = mp.spawn(
            save_episode_data,
            args=(
                shared_queue,
                total_episodes,
                dataset_export_dir_path,
                dataset_filename,
                env_name,
                all_done_event,
                True,
                1.0,
                validated_batch_splits,
            ),
            nprocs=1,
            join=False,
            daemon=False,
        )
        save_procs.extend(ctx_success.processes)

        # Signal all processes that they can exit
        lgr.info("Waiting all processes to exit...")
        all_done_event.wait()
        lgr.info("All processes signaled to exit, waiting for them to finish...")

        # Give processes time to exit cleanly
        time.sleep(5)

    except KeyboardInterrupt:
        lgr.info("\nProgram interrupted by user. Exiting...")
        # Terminate all processes
        for p in data_gen_procs + save_procs:
            if p.is_alive():
                p.terminate()
    finally:
        # Only close the main process's simulation app after all child processes have finished
        lgr.info("All processes completed, closing main simulation app...")
        simulation_app.close()
