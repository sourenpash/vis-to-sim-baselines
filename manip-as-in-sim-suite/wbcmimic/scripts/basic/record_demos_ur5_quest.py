# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Record demonstrations for Isaac Lab environments."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/dataset.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--step_hz", type=int, default=30, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
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
import inspect
import time

import gymnasium as gym
import numpy as np
import omni.log
import torch
from isaaclab.devices import Se3Keyboard
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim import RenderCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_mimic.utils.vr_policy import DualArmVRPolicy
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from scipy.spatial.transform import Rotation as R


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def goal_to_current_pose(goal_l, goal_r):
    goal_l = goal_l.cpu().numpy()
    goal_r = goal_r.cpu().numpy()
    return {
        "l": {
            "translation": goal_l[:3],
            "rotation": R.from_quat(goal_l[3:7], scalar_first=True).as_matrix(),
        },
        "r": {
            "translation": goal_r[:3],
            "rotation": R.from_quat(goal_r[3:7], scalar_first=True).as_matrix(),
        },
    }


def policy_actions_to_control_actions(
    policy_actions, cur_pose_l, cur_pose_r, device="cuda"
):
    if policy_actions["l"] is None:
        return cur_pose_l, cur_pose_r
    position_l = policy_actions["l"]["position"]
    rmat_l = policy_actions["l"]["rmat"]
    quat_l = R.from_matrix(rmat_l).as_quat(scalar_first=True)
    goal_l = torch.tensor(np.concatenate([position_l, quat_l]), device=device).reshape(
        7
    )
    position_r = policy_actions["r"]["position"]
    rmat_r = policy_actions["r"]["rmat"]
    quat_r = R.from_matrix(rmat_r).as_quat(scalar_first=True)
    goal_r = torch.tensor(np.concatenate([position_r, quat_r]), device=device).reshape(
        7
    )
    return goal_l, goal_r


def vr_policy_wrapper(current_poses, policy, side="r"):
    tmp_transform = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    # tmp_transform = np.eye(3)
    current_poses[side]["rotation"] = current_poses[side]["rotation"] @ tmp_transform
    vr_actions = policy.forward(current_poses)
    if vr_actions is None or vr_actions["l"] is None or vr_actions["r"] is None:
        return vr_actions
    vr_actions[side]["rmat"] = vr_actions[side]["rmat"] @ tmp_transform.T
    return vr_actions


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
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

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    env_cfg.observations.policy.concatenate_terms = False

    key_to_del = []
    for key in vars(env_cfg.observations).keys():
        if key not in ["policy", "images", "subtask_terms"]:
            key_to_del.append(key)
    for key in key_to_del:
        delattr(env_cfg.observations, key)

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.sim.device = "cpu"

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if inspect.isclass(success_term.func):
        success_term.func = success_term.func(env=env, cfg=env_cfg)
    # add teleoperation key for reset current recording instance
    should_reset_recording_instance = False
    success_manual = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def continue_recording():
        nonlocal should_reset_recording_instance
        print("continue_recording")
        should_reset_recording_instance = False

    def success_manual_callback():
        nonlocal success_manual
        success_manual = True

    # create controller
    keyboard_interface = Se3Keyboard(pos_sensitivity=0.2, rot_sensitivity=0.5)

    keyboard_interface.add_callback("R", reset_recording_instance)
    keyboard_interface.add_callback("S", success_manual_callback)
    keyboard_interface.add_callback("C", continue_recording)
    print(keyboard_interface)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker_r = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current_r")
    )
    goal_marker_r = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_r")
    )

    root_entity_cfg = SceneEntityCfg("robot", joint_names=[], body_names=["world"])
    root_entity_cfg.resolve(env.scene)
    policy = DualArmVRPolicy()
    # reset before starting
    env.reset()
    if isinstance(success_term.func, ManagerTermBase):
        success_term.func.reset()
    keyboard_interface.reset()
    robot = env.scene["robot"]
    ee_reset_pose_l = torch.tensor([0.16, 0.50, -0.03, 1, 0, 0, 0], device=env.device)
    ee_reset_pose_r = torch.cat(
        [
            env.observation_manager._obs_buffer["policy"]["eef_pos"][0].clone(),
            env.observation_manager._obs_buffer["policy"]["eef_quat"][0].clone(),
        ]
    )
    goal_l = ee_reset_pose_l.clone()
    goal_r = ee_reset_pose_r.clone()

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # get keyboard command
            # convert to torch
            # compute actions based on environment
            print("=" * 80)
            current_poses = goal_to_current_pose(goal_l, goal_r)
            vr_actions = vr_policy_wrapper(current_poses, policy)
            goal_l, goal_r = policy_actions_to_control_actions(
                vr_actions, goal_l, goal_r, device=env.device
            )
            if vr_actions is None or vr_actions["l"] is None or vr_actions["r"] is None:
                print("vr_actions is None")
                continue
            if vr_actions["l"]["reset"]:
                goal_l = ee_reset_pose_l.clone()
            if vr_actions["r"]["reset"]:
                goal_r = ee_reset_pose_r.clone()
            actions = torch.zeros_like(env.action_manager.action, device=env.device)
            root_pose_w = robot.data.body_state_w[:, root_entity_cfg.body_ids[0], 0:7]
            actions[:, :7] = torch.tensor(goal_r, device=env.device)
            actions[:, 7] = vr_actions["r"]["gripper"] < 0.022
            # perform action on environment
            obs, reward, terminated, truncated, info = env.step(actions)
            # print subtask_terms
            for k, v in obs["subtask_terms"].items():
                print(f"{k}: {v}")
            # success_term = None  # HACK: disable success check for now
            if success_term is not None or success_manual:
                if success_manual or bool(
                    success_term.func(env, **success_term.params)[0]
                ):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        env.recorder_manager.record_pre_reset(
                            [0], force_export_or_skip=False
                        )
                        env.recorder_manager.set_success_to_episodes(
                            [0],
                            torch.tensor([[True]], dtype=torch.bool, device=env.device),
                        )
                        env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0

            if should_reset_recording_instance:
                env.recorder_manager.reset()
                env.reset()
                if isinstance(success_term.func, ManagerTermBase):
                    success_term.func.reset()
                success_manual = False
                success_step_count = 0
                goal_l = ee_reset_pose_l.clone()
                goal_r = ee_reset_pose_r.clone()

                while should_reset_recording_instance:
                    actions = torch.zeros_like(
                        env.action_manager.action, device=env.device
                    )
                    root_pose_w = robot.data.body_state_w[
                        :, root_entity_cfg.body_ids[0], 0:7
                    ]
                    actions[0, :] = torch.cat([goal_r, torch.tensor([0])])
                    start_time = time.time()
                    env.step(actions)
                    end_time = time.time()
                    print(f"Time to step is {end_time - start_time}")

                env.recorder_manager.reset()
                env.reset()
                if isinstance(success_term.func, ManagerTermBase):
                    success_term.func.reset()
            # print out the current demo count if it has changed
            if (
                env.recorder_manager.exported_successful_episode_count
                > current_recorded_demo_count
            ):
                current_recorded_demo_count = (
                    env.recorder_manager.exported_successful_episode_count
                )
                print(
                    f"Recorded {current_recorded_demo_count} successful demonstrations."
                )

            if (
                args_cli.num_demos > 0
                and env.recorder_manager.exported_successful_episode_count
                >= args_cli.num_demos
            ):
                print(
                    f"All {args_cli.num_demos} demonstrations recorded. Exiting the app."
                )
                break

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

            # visualize
            eef_pose_w_r = torch.cat(
                combine_frame_transforms(
                    root_pose_w[0, 0:3].to(torch.float),
                    root_pose_w[0, 3:7].to(torch.float),
                    obs["policy"]["eef_pos"][0].to(torch.float),
                    obs["policy"]["eef_quat"][0].to(torch.float),
                )
            )
            ee_marker_r.visualize(eef_pose_w_r[None, 0:3], eef_pose_w_r[None, 3:7])
            goal_r_w = torch.cat(
                combine_frame_transforms(
                    root_pose_w[0, 0:3].to(torch.float),
                    root_pose_w[0, 3:7].to(torch.float),
                    goal_r[0:3].to(torch.float),
                    goal_r[3:7].to(torch.float),
                )
            )
            goal_marker_r.visualize(
                goal_r_w[None, 0:3],
                goal_r_w[None, 3:7],
            )

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
