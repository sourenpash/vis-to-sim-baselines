# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

# Launching Isaac Sim Simulator first.

import argparse
import json
import os
import time

import pinocchio as pin
from isaaclab.app import AppLauncher
from loguru import logger as lgr

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
    help="Choose control mode: origin, wbc (default: wbc) in mimicgen",
)
parser.add_argument(
    "--extra_wbc_params_path",
    type=str,
    default=None,
    help="Path to extra WBC parameters file. Default is None, which means no extra parameters are used.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import random

import carb
import gymnasium as gym
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.tasks
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_mimic.datagen.generation import (
    env_loop,
    setup_async_generation,
    setup_env_config,
)
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths


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
    return env_cfg


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name, output_failed_file_name, ext = setup_output_paths(
        args_cli.output_file
    )
    env_name = args_cli.task or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term, termination_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        ext=ext,
        state_only=args_cli.state_only,
        record_all=args_cli.record_all,
        remove_terminations=False,
    )
    env_cfg.seed = args_cli.seed

    # Parse extra WBC parameters if provided
    env_cfg = parser_wbc_params(args_cli.extra_wbc_params_path, env_cfg)

    # Add args_cli to env_cfg
    env_cfg.args_cli = args_cli

    # create environment
    env = gym.make(env_name, cfg=env_cfg)

    # Setup WBC if needed
    if args_cli.mimic_mode != "origin":
        from isaaclab_mimic.utils.mp_wbc import setup_wbc_mp_env

        if not hasattr(env.unwrapped, "setup_mp"):
            lgr.critical("Error: env must have setup_mp method.")
            assert False
        setup_wbc_mp_env(env.unwrapped)

    # set seed for generation
    random.seed(env.unwrapped.cfg.seed)
    np.random.seed(env.unwrapped.cfg.seed)
    torch.manual_seed(env.unwrapped.cfg.seed)

    # reset before starting
    env.reset()

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env.unwrapped,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        termination_term=termination_term,
        pause_subtask=args_cli.pause_subtask,
        mimic_mode=args_cli.mimic_mode,
        use_mp=True if args_cli.mimic_mode != "origin" else False,
    )

    try:
        asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env.unwrapped,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        lgr.trace("Tasks were cancelled.")

    lgr.info("Data generation task is done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
