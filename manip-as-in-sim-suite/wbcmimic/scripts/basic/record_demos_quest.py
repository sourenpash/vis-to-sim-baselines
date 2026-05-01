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

import pinocchio as pin
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
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import omni.log
import torch
from isaaclab.devices import Se3HandTracking, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ManagerBasedRLEnv, ViewerCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab_mimic.tasks.manager_based.arx7_sim.x7_pick_wine_bottle import (
    X7PickWineCfg,
)
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


def main():
    """Collect demonstrations from the environment using teleop interfaces."""

    rate_limiter = RateLimiter(args_cli.step_hz)

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    # Get USD path from environment variable or use relative path
    import os

    usd_path = os.environ.get(
        "ROOM_USD_PATH",
        os.path.join(
            os.path.dirname(__file__), "../../../assets/Room_empty_table.usdc"
        ),
    )
    env_cfg = X7PickWineCfg(usd_path=usd_path)
    env_cfg.env_name = "Isaac-X7-PickWine-Mimic-v0"

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

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

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
    ee_marker_l = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current_l")
    )
    ee_marker_r = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current_r")
    )
    goal_marker_l = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_l")
    )
    goal_marker_r = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_r")
    )

    root_entity_cfg = SceneEntityCfg("robot", joint_names=[], body_names=["base_link"])
    root_entity_cfg.resolve(env.scene)
    policy = DualArmVRPolicy()
    ee_reset_pose_l = torch.tensor([0.16, 0.50, -0.03, 1, 0, 0, 0], device=env.device)
    ee_reset_pose_r = torch.tensor([0.16, -0.50, -0.03, 1, 0, 0, 0], device=env.device)
    goal_l = ee_reset_pose_l.clone()
    goal_r = ee_reset_pose_r.clone()
    lift_des = 0.35

    # reset before starting
    env.reset()
    keyboard_interface.reset()
    robot = env.scene["robot"]

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    success_step_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # get keyboard command
            # convert to torch
            # compute actions based on environment
            current_poses = goal_to_current_pose(goal_l, goal_r)
            vr_actions = policy.forward(current_poses)
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
            assert actions.shape[-1] == 22
            lift_des = np.clip(lift_des + vr_actions["lift_speed"] * 0.01, 0.0, 0.6)
            actions[:, 0] = lift_des
            actions[:, -3] = vr_actions["forward_speed"]
            actions[:, -2] = vr_actions["side_speed"]
            actions[:, -1] = vr_actions["yaw_speed"]
            actions[:, 3:10] = torch.tensor(goal_l, device=env.device)
            actions[:, 10:17] = torch.tensor(goal_r, device=env.device)
            actions[:, 5] = actions[:, 5] + lift_des
            actions[:, 12] = actions[:, 12] + lift_des
            actions[:, 17] = vr_actions["l"]["gripper"]
            actions[:, 18] = vr_actions["r"]["gripper"]
            # perform action on environment
            obs, reward, terminated, truncated, info = env.step(actions)

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
                    assert actions.shape[-1] == 22
                    lift_des = np.clip(
                        lift_des + vr_actions["lift_speed"] * 0.01, 0.0, 0.6
                    )
                    actions[:, 0] = lift_des
                    actions[:, -3] = 0.0
                    actions[:, -2] = 0.0
                    actions[:, -1] = 0.0
                    actions[:, 3:10] = torch.tensor(goal_l, device=env.device)
                    actions[:, 10:17] = torch.tensor(goal_r, device=env.device)
                    actions[:, 5] = actions[:, 5] + lift_des
                    actions[:, 12] = actions[:, 12] + lift_des
                    actions[:, 17] = 0.044
                    actions[:, 18] = 0.044
                    env.step(actions)

                env.recorder_manager.reset()
                env.reset()

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
            eef_pose_w_l = torch.cat(
                combine_frame_transforms(
                    root_pose_w[0, 0:3].to(torch.float),
                    root_pose_w[0, 3:7].to(torch.float),
                    obs["policy"]["eef_pos_l"][0].to(torch.float),
                    obs["policy"]["eef_quat_l"][0].to(torch.float),
                )
            )
            eef_pose_w_r = torch.cat(
                combine_frame_transforms(
                    root_pose_w[0, 0:3].to(torch.float),
                    root_pose_w[0, 3:7].to(torch.float),
                    obs["policy"]["eef_pos_r"][0].to(torch.float),
                    obs["policy"]["eef_quat_r"][0].to(torch.float),
                )
            )
            ee_marker_l.visualize(eef_pose_w_l[None, 0:3], eef_pose_w_l[None, 3:7])
            ee_marker_r.visualize(eef_pose_w_r[None, 0:3], eef_pose_w_r[None, 3:7])
            goal_l_w = torch.cat(
                combine_frame_transforms(
                    root_pose_w[0, 0:3].to(torch.float),
                    root_pose_w[0, 3:7].to(torch.float),
                    goal_l[0:3].to(torch.float),
                    goal_l[3:7].to(torch.float),
                )
            )
            goal_r_w = torch.cat(
                combine_frame_transforms(
                    root_pose_w[0, 0:3].to(torch.float),
                    root_pose_w[0, 3:7].to(torch.float),
                    goal_r[0:3].to(torch.float),
                    goal_r[3:7].to(torch.float),
                )
            )
            goal_marker_l.visualize(
                goal_l_w[None, 0:3]
                + torch.tensor([0.0, 0.0, lift_des], device=env.device),
                goal_l_w[None, 3:7],
            )
            goal_marker_r.visualize(
                goal_r_w[None, 0:3]
                + torch.tensor([0.0, 0.0, lift_des], device=env.device),
                goal_r_w[None, 3:7],
            )

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
