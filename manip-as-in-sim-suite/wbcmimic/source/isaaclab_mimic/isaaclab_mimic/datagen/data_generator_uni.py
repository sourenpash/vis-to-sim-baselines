# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect

import isaaclab.utils.math as PoseUtils
import numpy as np
import torch
import torch.multiprocessing as mp
from isaaclab.envs import (
    ManagerBasedRLMimicEnv,
    MimicEnvCfg,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab_mimic.datagen.datagen_info import DatagenInfo
from isaaclab_mimic.datagen.selection_strategy import make_selection_strategy
from isaaclab_mimic.datagen.waypoint import (
    MultiWaypoint,
    MultiWaypointWbc,
    Waypoint,
    WaypointSequence,
    WaypointTrajectory,
)
from isaaclab_mimic.envs.wbc_env_config import (
    KeyPointSubTaskConfig,
    RLWbcSubTaskConfig,
    SubtaskControlMode,
    WbcSubTaskConfig,
)
from isaaclab_mimic.utils.exceptions import CollisionError, WbcStepOverMaxStepException
from loguru import logger as lgr

from .data_generator import DataGenerator
from .datagen_info_pool import DataGenInfoPool


class DataGeneratorWbc(DataGenerator):
    def __init__(
        self, env, src_demo_datagen_info_pool=None, dataset_path=None, demo_keys=None
    ):
        super().__init__(env, src_demo_datagen_info_pool, dataset_path, demo_keys)
        self.current_eef_subtask_control_mode = {}

    async def generate(
        self,
        env_id,
        success_term,
        termination_term,
        env_reset_queue=None,
        env_action_queue=None,
        pause_subtask=False,
        export_demo=True,
        use_mp=False,
    ):
        """
        Not consider constraints for rl and key-point now.
        """
        try:
            # reset the env to create a new task demo instance
            env_id_tensor = torch.tensor(
                [env_id], dtype=torch.int64, device=self.env.device
            )
            self.env.recorder_manager.reset(env_ids=env_id_tensor)
            success_term.func(self.env, **success_term.params)
            termination_term.func(self.env, **termination_term.params)
            self.env.reset(env_ids=env_id_tensor)
            for _ in range(self.warmup_steps):
                await env_action_queue.put((env_id, self.default_actions[env_id]))
                await env_action_queue.join()
            self.env.recorder_manager.reset(env_ids=env_id_tensor)
            self.env.recorder_manager.record_pre_reset(env_ids=env_id_tensor)
            self.env.recorder_manager.record_post_reset(env_ids=env_id_tensor)
            if isinstance(success_term.func, ManagerTermBase):
                success_term.func.reset(env_id)

            new_initial_state = self.env.scene.get_state(is_relative=True)

            # create runtime subtask constraint rules from subtask constraint configs
            self.runtime_subtask_constraints_dict[env_id] = {}
            for subtask_constraint in self.env_cfg.task_constraint_configs:
                self.runtime_subtask_constraints_dict[env_id].update(
                    subtask_constraint.generate_runtime_subtask_constraints()
                )

            # save generated data in these variables
            generated_states = []
            generated_obs = []
            generated_actions = []
            generated_success = False

            # some eef-specific state variables used during generation
            self.current_eef_selected_src_demo_indices[env_id] = {}
            self.current_eef_subtask_trajectories[env_id] = {}
            self.current_eef_subtask_indices[env_id] = {}
            self.current_eef_subtask_control_mode[env_id] = {}
            self.current_eef_subtask_step_indices[env_id] = {}
            eef_subtasks_done = {}
            for eef_name in self.env_cfg.subtask_configs.keys():
                self.current_eef_selected_src_demo_indices[env_id][eef_name] = None
                # prev_eef_executed_traj[eef_name] = None  # type of list of Waypoint
                self.current_eef_subtask_trajectories[env_id][
                    eef_name
                ] = None  # type of list of Waypoint
                self.current_eef_subtask_indices[env_id][eef_name] = 0
                self.current_eef_subtask_control_mode[env_id][eef_name] = (
                    SubtaskControlMode.get_mode_by_subtask_config(
                        self.env_cfg.subtask_configs[eef_name][0]
                    )
                )
                self.current_eef_subtask_step_indices[env_id][eef_name] = None
                eef_subtasks_done[eef_name] = False

            prev_src_demo_datagen_info_pool_size = 0
            # While loop that runs per time step
            is_first_step = True
            while True:
                async with self.src_demo_datagen_info_pool.asyncio_lock:
                    if (
                        len(self.src_demo_datagen_info_pool.datagen_infos)
                        > prev_src_demo_datagen_info_pool_size
                    ):
                        # src_demo_datagen_info_pool at this point may be updated with new demos,
                        # so we need to update subtask boundaries again
                        self.randomized_subtask_boundaries[env_id] = (
                            self.randomize_subtask_boundaries()
                        )  # shape [N, S, 2], last dim is start and end action lengths
                        prev_src_demo_datagen_info_pool_size = len(
                            self.src_demo_datagen_info_pool.datagen_infos
                        )

                    # generate trajectory for a subtask for the eef that is currently at the beginning of a subtask
                    for (
                        eef_name,
                        eef_subtask_step_index,
                    ) in self.current_eef_subtask_step_indices[env_id].items():
                        if eef_subtask_step_index is None:
                            if (
                                self.current_eef_subtask_control_mode[env_id][eef_name]
                                == SubtaskControlMode.WBC
                            ):
                                # self.current_eef_selected_src_demo_indices[env_id] will be updated in generate_trajectory
                                subtask_trajectory = self.generate_trajectory(
                                    env_id,
                                    eef_name,
                                    self.current_eef_subtask_indices[env_id][eef_name],
                                    self.randomized_subtask_boundaries[env_id],
                                    self.runtime_subtask_constraints_dict[env_id],
                                    self.current_eef_selected_src_demo_indices[env_id],
                                    self.current_eef_subtask_trajectories[env_id],
                                    wbc=True,
                                )
                                self.current_eef_subtask_trajectories[env_id][
                                    eef_name
                                ] = subtask_trajectory
                                self.current_eef_subtask_step_indices[env_id][
                                    eef_name
                                ] = 0
                                # self.current_eef_selected_src_demo_indices[env_id][eef_name] = selected_src_demo_inds
                                # two_arm_trajectories[task_spec_idx] = subtask_trajectory
                                # prev_executed_traj[task_spec_idx] = subtask_trajectory
                            elif (
                                self.current_eef_subtask_control_mode[env_id][eef_name]
                                == SubtaskControlMode.RL
                            ):
                                self.current_eef_subtask_step_indices[env_id][
                                    eef_name
                                ] = 0
                            elif (
                                self.current_eef_subtask_control_mode[env_id][eef_name]
                                == SubtaskControlMode.KEY_POINT
                            ):
                                subtask_cfg: KeyPointSubTaskConfig = (
                                    self.env_cfg.subtask_configs[eef_name][
                                        self.current_eef_subtask_indices[env_id][
                                            eef_name
                                        ]
                                    ]
                                )
                                traj_to_execute = WaypointTrajectory()
                                transformed_eef_poses = []
                                src_subtask_gripper_actions = []
                                robot_pos = (
                                    self.env.scene["robot"].data.root_pos_w
                                    - self.env.scene.env_origins
                                )
                                robot_quat = self.env.scene["robot"].data.root_quat_w
                                robot_pos = robot_pos[env_id]
                                robot_quat = robot_quat[env_id]
                                for i in range(len(subtask_cfg.key_eef_list)):
                                    eef_pose_in_cfg = torch.tensor(
                                        subtask_cfg.key_eef_list[i],
                                        device=self.env.device,
                                    )
                                    eef_pos_transformed, eef_quat_transformed = (
                                        PoseUtils.combine_frame_transforms(
                                            robot_pos,
                                            robot_quat,
                                            eef_pose_in_cfg[:3],
                                            eef_pose_in_cfg[3:],
                                        )
                                    )
                                    transformed_eef_poses.append(
                                        PoseUtils.make_pose(
                                            eef_pos_transformed,
                                            PoseUtils.matrix_from_quat(
                                                eef_quat_transformed
                                            ),
                                        )
                                    )
                                    src_subtask_gripper_actions.append(
                                        subtask_cfg.key_gripper_action_list[i]
                                    )

                                transformed_eef_poses = torch.stack(
                                    transformed_eef_poses, dim=0
                                )
                                src_subtask_gripper_actions = torch.tensor(
                                    src_subtask_gripper_actions, device=self.env.device
                                )
                                traj_to_execute.add_waypoint_sequence(
                                    WaypointSequence.from_poses(
                                        poses=transformed_eef_poses,
                                        gripper_actions=src_subtask_gripper_actions,
                                        action_noise=0.0,
                                    )
                                )
                                subtask_trajectory = (
                                    traj_to_execute.get_full_sequence().sequence
                                )
                                self.current_eef_subtask_trajectories[env_id][
                                    eef_name
                                ] = subtask_trajectory
                                self.current_eef_subtask_step_indices[env_id][
                                    eef_name
                                ] = 0

                # determine the next waypoint for each eef based on the current subtask constraints
                eef_waypoint_dict = {}
                for eef_name in sorted(self.env_cfg.subtask_configs.keys()):
                    # handle constraints
                    step_ind = self.current_eef_subtask_step_indices[env_id][eef_name]
                    subtask_ind = self.current_eef_subtask_indices[env_id][eef_name]
                    if (eef_name, subtask_ind) in self.runtime_subtask_constraints_dict[
                        env_id
                    ]:
                        task_constraint = self.runtime_subtask_constraints_dict[env_id][
                            (eef_name, subtask_ind)
                        ]
                        if (
                            task_constraint["type"]
                            == SubTaskConstraintType._SEQUENTIAL_LATTER
                        ):
                            min_time_diff = task_constraint["min_time_diff"]
                            if not task_constraint["fulfilled"]:
                                if (
                                    min_time_diff == -1
                                    or step_ind
                                    >= len(
                                        self.current_eef_subtask_trajectories[env_id][
                                            eef_name
                                        ]
                                    )
                                    - min_time_diff
                                ):
                                    if step_ind > 0:
                                        # wait at the same step
                                        step_ind -= 1
                                        self.current_eef_subtask_step_indices[env_id][
                                            eef_name
                                        ] = step_ind

                        elif (
                            task_constraint["type"]
                            == SubTaskConstraintType.COORDINATION
                        ):
                            synchronous_steps = task_constraint["synchronous_steps"]
                            concurrent_task_spec_key = task_constraint[
                                "concurrent_task_spec_key"
                            ]
                            concurrent_subtask_ind = task_constraint[
                                "concurrent_subtask_ind"
                            ]
                            concurrent_task_fulfilled = (
                                self.runtime_subtask_constraints_dict[env_id][
                                    (concurrent_task_spec_key, concurrent_subtask_ind)
                                ]["fulfilled"]
                            )

                            if (
                                task_constraint["coordination_synchronize_start"]
                                and self.current_eef_subtask_indices[env_id][
                                    concurrent_task_spec_key
                                ]
                                < concurrent_subtask_ind
                            ):
                                # the concurrent eef is not yet at the concurrent subtask, so wait at the first action
                                # this also makes sure that the concurrent task starts at the same time as this task
                                step_ind = 0
                                self.current_eef_subtask_step_indices[env_id][
                                    eef_name
                                ] = 0
                            else:
                                if (
                                    not concurrent_task_fulfilled
                                    and step_ind
                                    >= len(
                                        self.current_eef_subtask_trajectories[env_id][
                                            eef_name
                                        ]
                                    )
                                    - synchronous_steps
                                ):
                                    # trigger concurrent task
                                    self.runtime_subtask_constraints_dict[env_id][
                                        (
                                            concurrent_task_spec_key,
                                            concurrent_subtask_ind,
                                        )
                                    ]["fulfilled"] = True

                                if not task_constraint["fulfilled"]:
                                    if (
                                        step_ind
                                        >= len(
                                            self.current_eef_subtask_trajectories[
                                                env_id
                                            ][eef_name]
                                        )
                                        - synchronous_steps
                                    ):
                                        if step_ind > 0:
                                            step_ind -= 1
                                            self.current_eef_subtask_step_indices[
                                                env_id
                                            ][
                                                eef_name
                                            ] = step_ind  # wait here

                    waypoint = self.current_eef_subtask_trajectories[env_id][eef_name][
                        step_ind
                    ]
                    eef_waypoint_dict[eef_name] = waypoint
                multi_waypoint = MultiWaypointWbc(eef_waypoint_dict)

                # TODO: only support one eef for now

                if (
                    self.current_eef_subtask_control_mode[env_id][eef_name]
                    == SubtaskControlMode.WBC
                    or self.current_eef_subtask_control_mode[env_id][eef_name]
                    == SubtaskControlMode.KEY_POINT
                ):
                    # execute the next waypoints for all eefs
                    if is_first_step:
                        wbc_max_step = self.env_cfg.subtask_configs[eef_name][
                            subtask_ind
                        ].wbc_max_step_start
                    elif (
                        self.current_eef_subtask_step_indices[env_id][eef_name]
                        == len(self.current_eef_subtask_trajectories[env_id][eef_name])
                        - 1
                    ):
                        wbc_max_step = self.env_cfg.subtask_configs[eef_name][
                            subtask_ind
                        ].wbc_max_step_last
                    else:
                        wbc_max_step = self.env_cfg.subtask_configs[eef_name][
                            subtask_ind
                        ].wbc_max_step
                    try:
                        exec_results = await multi_waypoint.execute(
                            env=self.env,
                            success_term=success_term,
                            env_id=env_id,
                            env_action_queue=env_action_queue,
                            wbc_max_step=wbc_max_step,
                            vel_multiplier=getattr(
                                self.env_cfg.subtask_configs[eef_name][subtask_ind],
                                "vel_multiplier",
                                1.0,
                            ),
                            mp=use_mp,
                            success_break=(
                                self.current_eef_subtask_control_mode[env_id][eef_name]
                                == SubtaskControlMode.WBC
                            ),
                        )
                    except CollisionError as e:
                        lgr.error(str(e))
                        # collision error, reset the env
                        generated_success = False
                        break
                    except WbcStepOverMaxStepException as e:
                        # wbc step over max step, reset the env
                        lgr.warning("WBC step over max step, resetting the env")
                        generated_success = bool(
                            success_term.func(self.env, **success_term.params)[env_id]
                        )
                        break
                    is_first_step = False
                    # update execution state buffers
                    if len(exec_results["states"]) > 0:
                        generated_states.extend(exec_results["states"])
                        generated_obs.extend(exec_results["observations"])
                        generated_actions.extend(exec_results["actions"])
                        generated_success = generated_success or exec_results["success"]

                    subtask_break = False
                    subtask_success = False
                    for eef_name in self.env_cfg.subtask_configs.keys():
                        self.current_eef_subtask_step_indices[env_id][eef_name] += 1
                        subtask_ind = self.current_eef_subtask_indices[env_id][eef_name]

                        if self.current_eef_subtask_step_indices[env_id][eef_name] >= (
                            len(self.current_eef_subtask_trajectories[env_id][eef_name])
                            - self.env_cfg.subtask_configs[eef_name][
                                self.current_eef_subtask_indices[env_id][eef_name]
                            ].subtask_term_offset_range[1]
                            - 10
                        ):
                            subtask_term_name = self.env.subtasks[eef_name][
                                self.current_eef_subtask_indices[env_id][eef_name]
                            ]
                            subtask_success = (
                                subtask_success
                                or self.env.obs_buf["subtask_terms"][subtask_term_name][
                                    env_id
                                ]
                                if subtask_term_name is not None
                                else True
                            )

                        if self.current_eef_subtask_step_indices[env_id][
                            eef_name
                        ] == len(
                            self.current_eef_subtask_trajectories[env_id][eef_name]
                        ):  # subtask done
                            if (
                                eef_name,
                                subtask_ind,
                            ) in self.runtime_subtask_constraints_dict[env_id]:
                                task_constraint = self.runtime_subtask_constraints_dict[
                                    env_id
                                ][(eef_name, subtask_ind)]
                                if (
                                    task_constraint["type"]
                                    == SubTaskConstraintType._SEQUENTIAL_FORMER
                                ):
                                    constrained_task_spec_key = task_constraint[
                                        "constrained_task_spec_key"
                                    ]
                                    constrained_subtask_ind = task_constraint[
                                        "constrained_subtask_ind"
                                    ]
                                    self.runtime_subtask_constraints_dict[env_id][
                                        (
                                            constrained_task_spec_key,
                                            constrained_subtask_ind,
                                        )
                                    ]["fulfilled"] = True
                                elif (
                                    task_constraint["type"]
                                    == SubTaskConstraintType.COORDINATION
                                ):
                                    concurrent_task_spec_key = task_constraint[
                                        "concurrent_task_spec_key"
                                    ]
                                    concurrent_subtask_ind = task_constraint[
                                        "concurrent_subtask_ind"
                                    ]
                                    # concurrent_task_spec_idx = task_spec_keys.index(concurrent_task_spec_key)
                                    task_constraint["finished"] = True
                                    # check if concurrent task has been finished
                                    assert (
                                        self.runtime_subtask_constraints_dict[env_id][
                                            (
                                                concurrent_task_spec_key,
                                                concurrent_subtask_ind,
                                            )
                                        ]["finished"]
                                        or self.current_eef_subtask_step_indices[
                                            env_id
                                        ][concurrent_task_spec_key]
                                        >= len(
                                            self.current_eef_subtask_trajectories[
                                                env_id
                                            ][concurrent_task_spec_key]
                                        )
                                        - 1
                                    )

                            if pause_subtask:
                                input(
                                    f"Pausing after subtask {self.current_eef_subtask_indices[env_id][eef_name]} of {eef_name} execution."
                                    " Press any key to continue..."
                                )
                            subtask_term_name = self.env.subtasks[eef_name][
                                self.current_eef_subtask_indices[env_id][eef_name]
                            ]

                            # This is a check to see if this arm has completed all the subtasks
                            if (
                                self.current_eef_subtask_indices[env_id][eef_name]
                                == len(self.env_cfg.subtask_configs[eef_name]) - 1
                            ):
                                eef_subtasks_done[eef_name] = True
                                # If all subtasks done for this arm, repeat last waypoint to make sure this arm does not move
                                self.current_eef_subtask_trajectories[env_id][
                                    eef_name
                                ].append(
                                    self.current_eef_subtask_trajectories[env_id][
                                        eef_name
                                    ][-1]
                                )
                            else:
                                self.current_eef_subtask_step_indices[env_id][
                                    eef_name
                                ] = None
                                subtask_term_name = self.env.subtasks[eef_name][
                                    self.current_eef_subtask_indices[env_id][eef_name]
                                ]
                                if not (
                                    subtask_success
                                    or self.env.obs_buf["subtask_terms"][
                                        subtask_term_name
                                    ][env_id]
                                    if subtask_term_name is not None
                                    else True
                                ):
                                    lgr.info(
                                        f"Env {env_id}: Sub task {self.env.subtasks[eef_name][self.current_eef_subtask_indices[env_id][eef_name]]} done not not success"
                                    )
                                    generated_success = success_term.func(
                                        self.env, **success_term.params
                                    )[env_id]
                                    subtask_break = True

                                self.current_eef_subtask_indices[env_id][eef_name] += 1
                                self.current_eef_subtask_control_mode[env_id][
                                    eef_name
                                ] = SubtaskControlMode.get_mode_by_subtask_config(
                                    self.env_cfg.subtask_configs[eef_name][
                                        self.current_eef_subtask_indices[env_id][
                                            eef_name
                                        ]
                                    ]
                                )
                else:
                    raise NotImplementedError(
                        "RL control logic is not implemented in this generator."
                    )

                # Check if all eef_subtasks_done values are True
                if all(eef_subtasks_done.values()) or subtask_break:
                    lgr.info(
                        f"Env {env_id}",
                        all(eef_subtasks_done.values()),
                        "subtask break: ",
                        subtask_break,
                    )
                    break
                if (
                    torch.mean(
                        torch.abs(self.env.obs_buf["policy"]["joint_vel"][env_id])
                    )
                    > 10
                ):
                    lgr.warning("physics error detected, reset")
                    break

            # del self.current_eef_selected_src_demo_indices[env_id]
            # del self.current_eef_subtask_trajectories[env_id]
            # del self.current_eef_subtask_indices[env_id]
            # del self.current_eef_subtask_step_indices[env_id]

            # self.current_eef_selected_src_demo_indices[env_id] = {}
            # self.current_eef_subtask_trajectories[env_id] = {}
            # self.current_eef_subtask_indices[env_id] = {}
            # self.current_eef_subtask_step_indices[env_id] = {}

            # merge numpy arrays
            if len(generated_actions) > 0:
                generated_actions = torch.cat(generated_actions, dim=0)

            # set success to the recorded episode data and export to file
            self.env.recorder_manager.set_success_to_episodes(
                env_id_tensor,
                torch.tensor(
                    [[generated_success]], dtype=torch.bool, device=self.env.device
                ),
            )
            if export_demo:
                self.env.recorder_manager.export_episodes(env_id_tensor)

            results = dict(
                initial_state=new_initial_state,
                states=generated_states,
                observations=generated_obs,
                actions=generated_actions,
                success=generated_success,
            )
            return results
        except Exception as e:
            lgr.error(f"Error in generate: {e}")
            import traceback

            traceback.print_exc()
