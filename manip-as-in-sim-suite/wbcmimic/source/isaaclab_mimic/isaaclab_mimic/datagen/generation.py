# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import inspect
import time
import tracemalloc
from typing import Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLMimicEnv
from isaaclab.managers import DatasetExportMode, ManagerTermBase, TerminationTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.data_generator_uni import DataGeneratorWbc
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
from isaaclab_mimic.envs.mdp import (
    ActionStateImageDepthRecorderManagerCfg,
    ActionStateRecorderManagerCfg,
)
from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from loguru import logger as lgr

# global variable to keep track of the data generation statistics
num_success = 0
num_failures = 0
num_attempts = 0
INTERVAL = 400


async def run_data_generator(
    env: ManagerBasedRLMimicEnv,
    env_id: int,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term: TerminationTermCfg,
    termination_term: TerminationTermCfg,
    pause_subtask: bool = False,
    use_mp=False,
):
    """Run mimic data generation from the given data generator in the specified environment index.

    Args:
        env: The environment to run the data generator on.
        env_id: The environment index to run the data generation on.
        env_reset_queue: The asyncio queue to send environment (for this particular env_id) reset requests to.
        env_action_queue: The asyncio queue to send actions to for executing actions.
        data_generator: The data generator instance to use.
        success_term: The success termination term to use.
        pause_subtask: Whether to pause the subtask during generation.
    """
    global num_success, num_failures, num_attempts
    tracemalloc.start()
    old_snapshot = tracemalloc.take_snapshot()
    if inspect.isclass(success_term.func):
        success_term.func = success_term.func(cfg=success_term, env=env)
        env.success_term = success_term.func
    while True:
        results = await data_generator.generate(
            env_id=env_id,
            success_term=success_term,
            termination_term=termination_term,
            env_reset_queue=env_reset_queue,
            env_action_queue=env_action_queue,
            pause_subtask=pause_subtask,
            use_mp=use_mp,
        )
        if bool(results["success"]):
            num_success += 1
        else:
            num_failures += 1
        num_attempts += 1
        new_snapshot = tracemalloc.take_snapshot()
        top_stats = new_snapshot.compare_to(old_snapshot, "lineno")
        lgr.trace("[ Top 10 differences at data generation ]")
        for stat in top_stats[:10]:
            lgr.trace(str(stat))
        old_snapshot = new_snapshot


def env_loop(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    shared_datagen_info_pool: DataGenInfoPool,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    print_time: bool = False,
):
    """Main asyncio loop for the environment.

    Args:
        env: The environment to run the main step loop on.
        env_reset_queue: The asyncio queue to handle reset request the environment.
        env_action_queue: The asyncio queue to handle actions to for executing actions.
        shared_datagen_info_pool: The shared datagen info pool that stores source demo info.
        asyncio_event_loop: The main asyncio event loop.
    """
    global num_success, num_failures, num_attempts
    try:
        env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
        prev_num_attempts = 0

        robot: Articulation = env.unwrapped.scene["robot"]
        global_idx = 0
        start_time = time.perf_counter()
        with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
            while True:
                start_time_this = time.perf_counter()
                # check if any environment needs to be reset while waiting for actions
                if env_reset_queue is not None:
                    while env_action_queue.qsize() != env.num_envs:
                        asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                        while not env_reset_queue.empty():
                            env_id_tensor[0] = env_reset_queue.get_nowait()
                            env.reset(env_ids=env_id_tensor)
                            env_reset_queue.task_done()

                actions = torch.zeros(env.action_space.shape)
                if global_idx % INTERVAL == 0:
                    lgr.trace(
                        "Time before actions: ",
                        1000 * (time.perf_counter() - start_time_this),
                        "ms",
                    )
                start_time_tmp = time.perf_counter()
                # get actions from all the data generators
                for i in range(env.num_envs):
                    # an async-blocking call to get an action from a data generator
                    env_id, action = asyncio_event_loop.run_until_complete(
                        env_action_queue.get()
                    )
                    actions[env_id] = action
                if global_idx % INTERVAL == 0:
                    lgr.trace(
                        f"Compute action time: {1000 * (time.perf_counter() - start_time_tmp)} ms"
                    )
                # perform action on environment
                start_time_tmp = time.perf_counter()
                obs = env.step(actions)
                if global_idx % INTERVAL == 0:
                    lgr.trace(
                        f"Step time: {(time.perf_counter() - start_time_tmp) * 1000} ms"
                    )
                start_time_tmp = time.perf_counter()
                global_idx += 1
                actions = actions.to(robot.data.root_pos_w.device)

                # mark done so the data generators can continue with the step results
                for i in range(env.num_envs):
                    env_action_queue.task_done()

                if prev_num_attempts != num_attempts:
                    prev_num_attempts = num_attempts
                    generated_sucess_rate = (
                        100 * num_success / num_attempts if num_attempts > 0 else 0.0
                    )
                    lgr.info("")
                    lgr.info("*" * 50, "\033[K")
                    lgr.info(
                        f"{num_success}/{num_attempts} ({generated_sucess_rate:.1f}%) successful demos generated by"
                        f" mimic at time {time.time()}\033[K"
                    )
                    lgr.info("*" * 50, "\033[K")

                    # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
                    generation_guarantee = env.cfg.datagen_config.generation_guarantee
                    generation_num_trials = env.cfg.datagen_config.generation_num_trials
                    check_val = num_success if generation_guarantee else num_attempts
                    if check_val >= generation_num_trials:
                        lgr.info(
                            f"Reached {generation_num_trials} successes/attempts. Exiting."
                        )
                        break

                # check that simulation is stopped or not
                if env.sim.is_stopped():
                    break
                if global_idx % INTERVAL == 0:
                    lgr.trace(
                        "After step time: ",
                        1000 * (time.perf_counter() - start_time_tmp),
                        "ms",
                    )
                    lgr.trace(
                        f"Average FPS: {global_idx / (time.perf_counter() - start_time)}"
                    )
                    lgr.trace(f"FPS: {1 / (time.perf_counter() - start_time_this)}")
        env.close()
    except Exception as e:
        lgr.error(f"Exception in env loop: {e}")
        import traceback

        traceback.print_exc()


def setup_env_config(
    env_name: str,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_num_trials: int | None = None,
    dataset_file_handler_class_type=None,
    ext: str = ".hdf5",
    state_only=False,
    record_all=False,
    remove_terminations=True,
) -> tuple[Any, Any, Any]:
    """Configure the environment for data generation.

    Args:
        env_name: Name of the environment
        output_dir: Directory to save output
        output_file_name: Name of output file
        num_envs: Number of environments to run
        device: Device to run on
        generation_num_trials: Optional override for number of trials

    Returns:
        tuple containing:
            - env_cfg: The environment configuration
            - success_term: The success termination condition

    Raises:
        NotImplementedError: If no success termination term found
    """
    env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)

    if generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = generation_num_trials

    env_cfg.env_name = env_name

    # Extract success checking function
    success_term = None
    termination_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        termination_term = env_cfg.terminations.early_stop
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError(
            "No success termination term was found in the environment."
        )

    # Configure for data generation
    if remove_terminations:
        env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    # Setup recorders
    if state_only:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        if hasattr(env_cfg.observations, "images"):
            delattr(env_cfg.observations, "images")
        if hasattr(env_cfg.observations, "depths"):
            delattr(env_cfg.observations, "depths")
        if hasattr(env_cfg.observations, "policy_infer"):
            delattr(env_cfg.observations, "policy_infer")
        if hasattr(env_cfg.observations, "camera_info"):
            delattr(env_cfg.observations, "camera_info")
        item_to_delete = []
        for item in vars(env_cfg.scene).keys():
            if "camera" in item:
                item_to_delete.append(item)
        for item in item_to_delete:
            delattr(env_cfg.scene, item)
    else:
        env_cfg.recorders = ActionStateImageDepthRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    lgr.info("Save to {}".format(output_file_name + ext))

    if dataset_file_handler_class_type is not None:
        env_cfg.recorders.dataset_file_handler_class_type = (
            dataset_file_handler_class_type
        )
    else:
        if ext[-4:] == "zarr":
            env_cfg.recorders.dataset_file_handler_class_type = (
                ZarrCompatibleDatasetFileHandler
            )
        elif ext[-4:] == "hdf5":
            env_cfg.recorders.dataset_file_handler_class_type = HDF5DatasetFileHandler
        else:
            raise NotImplementedError(
                f"unsupport for file not end with 'hdf5' or 'zarr'."
            )

    if env_cfg.datagen_config.generation_keep_failed or record_all:
        env_cfg.recorders.dataset_export_mode = (
            DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        )
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term, termination_term


def setup_async_generation(
    env: Any,
    num_envs: int,
    input_file: str,
    success_term: Any,
    termination_term: Any,
    pause_subtask: bool = False,
    mimic_mode: str = "origin",
    use_mp: bool = False,
) -> dict[str, Any]:
    """Setup async data generation tasks.

    Args:
        env: The environment instance
        num_envs: Number of environments to run
        input_file: Path to input dataset file
        success_term: Success condition
        termination_term: Termination condition
        pause_subtask: Whether to pause after subtasks

    Returns:
        List of asyncio tasks for data generation
    """
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()
    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(
        env, env.cfg, env.device, asyncio_lock=shared_datagen_info_pool_lock
    )
    shared_datagen_info_pool.load_from_dataset_file(input_file)
    lgr.info(
        f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool"
    )

    # Create and schedule data generator tasks
    if mimic_mode == "origin":
        data_generator = DataGenerator(
            env=env, src_demo_datagen_info_pool=shared_datagen_info_pool
        )
    elif mimic_mode == "wbc":
        data_generator = DataGeneratorWbc(
            env=env, src_demo_datagen_info_pool=shared_datagen_info_pool
        )
    else:
        raise ValueError(f"Unknown mimic model: {mimic_mode}")
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        task = asyncio_event_loop.create_task(
            run_data_generator(
                env,
                i,
                env_reset_queue,
                env_action_queue,
                data_generator,
                success_term,
                termination_term,
                pause_subtask=pause_subtask,
                use_mp=use_mp,
            )
        )
        data_generator_asyncio_tasks.append(task)

    return {
        "tasks": data_generator_asyncio_tasks,
        "event_loop": asyncio_event_loop,
        "reset_queue": env_reset_queue,
        "action_queue": env_action_queue,
        "info_pool": shared_datagen_info_pool,
    }
