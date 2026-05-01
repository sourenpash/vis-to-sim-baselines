# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import time
from multiprocessing.connection import Connection

import numpy as np
import torch.multiprocessing as mp
from loguru import logger as lgr


def run(rank, wbc_solver_cfg, wbc_pipe, env_id, step_dt, args_cli):
    try:
        from isaaclab_mimic.utils.exceptions import WbcSolveFailedException
        from isaaclab_mimic.utils.robots.wbc_controller import WbcController

        _wbc_controller = WbcController(wbc_solver_cfg)
        _step_dt = step_dt
        _pipe = wbc_pipe
        _env_id = env_id
        while True:
            q0, robot_world_pose_np, target_eef_pose_dict_send, vel_multiplier = (
                _pipe.recv()
            )
            start_time = time.perf_counter()
            _wbc_controller.update_joint_pos(q0)
            _wbc_controller.update_root_pose(robot_world_pose_np)

            for ee, target_eef_pose in target_eef_pose_dict_send.items():
                _wbc_controller.set_goal(target_eef_pose)

            try:
                success, qd = _wbc_controller.step_robot(
                    vel_multiplier=vel_multiplier
                    # distance=d,
                    # norm_a2b=d_vec,
                    # index_a=np.arange(0, 22),
                    # body_names=robot_entity_cfg.body_names
                )
            except WbcSolveFailedException:
                lgr.warning("WbcSolveFailedException at env {}".format(_env_id))
                success = True
                qd = np.zeros_like(q0)

            q0 = q0 + qd * _step_dt
            _pipe.send((success, q0))
            end_time = time.perf_counter()
    except Exception as e:
        lgr.error(str(e))
        import traceback

        traceback.print_exc()


def run_dualarm_wbc(rank, wbc_solver_cfg, wbc_pipe, env_id, step_dt, args_cli):
    try:
        from isaaclab_mimic.utils.exceptions import WbcSolveFailedException
        from isaaclab_mimic.utils.robots.wbc_controller_dual import DualArmWbcController

        _wbc_controller = DualArmWbcController(wbc_solver_cfg)
        _step_dt = step_dt
        _pipe = wbc_pipe
        _env_id = env_id
        while True:
            q0, robot_world_pose_np, target_eef_pose_dict_send = _pipe.recv()
            _wbc_controller.update_joint_pos(q0)
            _wbc_controller.update_root_pose(robot_world_pose_np)
            for ee, target_eef_pose in target_eef_pose_dict_send.items():
                _wbc_controller.set_goal(ee, target_eef_pose)

            try:
                success, qd = _wbc_controller.step_robot(
                    # distance=d,
                    # norm_a2b=d_vec,
                    # index_a=np.arange(0, 22),
                    # body_names=robot_entity_cfg.body_names
                )
            except WbcSolveFailedException:
                lgr.warning("WbcSolveFailedException at env {}".format(_env_id))
                success = True
                qd = np.zeros_like(q0)
            q0 = q0 + qd * _step_dt
            q0[:3] = qd[:3]
            _pipe.send((success, q0))
    except Exception as e:
        lgr.error(str(e))
        import traceback

        traceback.print_exc()


def setup_wbc_mp_env(env):
    from isaaclab_mimic.utils.robots.wbc_controller import WbcControllerCfg
    from isaaclab_mimic.utils.robots.wbc_controller_dual import DualArmWbcControllerCfg

    wbc_solvers_process = []
    wbc_solver_pipes: list[Connection] = []
    for env_id in range(env.cfg.scene.num_envs):
        father_end, child_end = mp.Pipe()
        try:
            if isinstance(env.cfg.mimic_config.wbc_solver_cfg, WbcControllerCfg):
                process = mp.Process(
                    target=run,
                    args=(
                        0,
                        env.cfg.mimic_config.wbc_solver_cfg,
                        child_end,
                        env_id,
                        env.cfg.sim.dt * env.cfg.decimation,
                        env.cfg.args_cli,
                    ),
                )
            elif isinstance(
                env.cfg.mimic_config.wbc_solver_cfg, DualArmWbcControllerCfg
            ):
                process = mp.Process(
                    target=run_dualarm_wbc,
                    args=(
                        0,
                        env.cfg.mimic_config.wbc_solver_cfg,
                        child_end,
                        env_id,
                        env.cfg.sim.dt * env.cfg.decimation,
                        env.cfg.args_cli,
                    ),
                )
            process.start()
        except Exception as e:
            lgr.error(str(e))
            import traceback

            traceback.print_exc()
        wbc_solvers_process.append(process)
        wbc_solver_pipes.append(father_end)
    env.setup_mp(wbc_solvers_process, wbc_solver_pipes)
