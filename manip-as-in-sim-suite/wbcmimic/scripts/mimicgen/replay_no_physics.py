# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
脚本功能：从数据集中读取状态并设置到环境中，关闭物理仿真但保留渲染
主要用途：通过replay重现可视化，但不运行物理仿真

Script to replay states from dataset with rendering but WITHOUT physics simulation.
Main use: Visualize replay data without running physics.
"""

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

# 添加命令行参数
parser = argparse.ArgumentParser(
    description="Replay states from dataset with rendering but without physics simulation."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="并行环境数量 / Number of parallel environments",
)
parser.add_argument(
    "--task", type=str, required=True, help="任务名称 / Name of the task"
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="输入数据集文件路径 / Input dataset file path",
)
parser.add_argument(
    "--episode_indices",
    type=int,
    nargs="+",
    default=None,
    help="要处理的episode索引列表，默认处理所有 / Episode indices to process",
)
parser.add_argument(
    "--render_interval",
    type=int,
    default=1,
    help="渲染间隔，每N步渲染一次 / Render every N steps",
)
parser.add_argument(
    "--replay_speed",
    type=float,
    default=1.0,
    help="回放速度倍率 / Replay speed multiplier",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="使用多GPU并行运行 / Run with multiple GPUs",
)
parser.add_argument(
    "--n_procs", type=int, default=1, help="进程数量 / Number of processes"
)
parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help="输出数据集文件路径（如果提供，将录制重放）/ Output dataset file path (if provided, will record replay)",
)
parser.add_argument(
    "--record_images",
    action="store_true",
    default=False,
    help="是否录制图像数据 / Whether to record image data",
)
parser.add_argument(
    "--max_episodes_num",
    type=int,
    default=-1,
    help="最大处理的episode数量，-1表示不限制 / Maximum number of episodes to process, -1 means no limit",
)


# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()


def concatenate_state(state_lst):
    """
    连接多个环境的状态
    Concatenate states from multiple environments
    """
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


def compare_states(state_from_dataset, runtime_state, device, num_envs):
    """
    比较数据集状态和运行时状态
    Compare states from dataset and runtime

    Args:
        state_from_dataset: 来自数据集的状态 / State from dataset
        runtime_state: 当前运行时状态 / Current runtime state
        device: 设备类型 / Device type
        num_envs: 环境数量 / Number of environments

    Returns:
        torch.Tensor: 每个环境的状态匹配结果 / State match results for each environment
    """
    states_matched = torch.tensor([True] * num_envs, device=device)

    # 遍历所有资产类型（articulation和rigid_object）
    for asset_type in ["articulation", "rigid_object"]:
        if asset_type not in runtime_state:
            continue

        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][
                    state_name
                ]

                # 检查状态维度是否匹配
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(
                        f"State shape of {state_name} for asset {asset_name} don't match"
                    )

                # 比较状态值（阈值设为0.01）
                states_matched = torch.logical_and(
                    states_matched,
                    (
                        abs(dataset_asset_state.to(device) - runtime_asset_state) < 0.01
                    ).all(dim=1),
                )

    return states_matched


def save_episode_data(
    rank,
    episode_data_queue,
    total_episodes,
    dataset_export_dir_path,
    dataset_filename,
    env_name,
    stop_queue,
):
    """
    保存episode数据的进程函数
    Process function to save episode data
    """
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)

    print(f">>>保存episode数据进程 / Save episode data process rank: {rank}")

    app_launcher = AppLauncher(args_cli)
    from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler

    dataset_file_handler = ZarrCompatibleDatasetFileHandler()
    dataset_file_handler.create(
        os.path.join(dataset_export_dir_path, dataset_filename), env_name=env_name
    )

    idx = 0
    while idx < total_episodes:
        episode_data = episode_data_queue.get()
        dataset_file_handler.write_episode(episode_data)
        dataset_file_handler.flush()
        idx += 1
        episode_data_queue.task_done()
        print(
            f"完成写入第 {idx} 个episode / Finished writing episode {idx}", flush=True
        )
        if idx == args_cli.max_episodes_num:
            break

    dataset_file_handler.close()
    for i in range(args_cli.n_procs):
        stop_queue.put(1)
    time.sleep(10)


def replay_worker(
    rank, world_size, episode_data_queue, episode_indices_to_replay_lst, stop_queue
):
    """
    工作进程函数，处理分配给它的episodes
    Worker process function to handle assigned episodes
    """
    try:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        print(f">>>工作进程 / Worker process rank: {rank}")

        # 启动模拟器（带渲染模式）
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        # 导入必要的模块
        import isaaclab_mimic.envs  # noqa: F401
        import isaaclab_mimic.tasks  # noqa: F401
        import isaaclab_mimic.utils.tensor_utils as TensorUtils
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.datasets import EpisodeData
        from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # 加载输入数据集
        print(f"[Rank {rank}] 加载数据集 / Loading dataset: {args_cli.input_file}")
        if not os.path.exists(args_cli.input_file):
            raise FileNotFoundError(
                f"输入数据集文件不存在 / Input dataset file not found: {args_cli.input_file}"
            )

        dataset_file_handler = ZarrCompatibleDatasetFileHandler()
        dataset_file_handler.open(args_cli.input_file)

        # 获取数据集信息
        env_name = args_cli.task or dataset_file_handler.get_env_name()
        episode_count = dataset_file_handler.get_num_episodes()
        episode_names = list(dataset_file_handler.get_episode_names())

        print(
            f"[Rank {rank}] 数据集包含 {episode_count} 个episodes / Dataset contains {episode_count} episodes"
        )

        if episode_count == 0:
            print("数据集中没有找到episodes / No episodes found in dataset")
            return

        # 解析环境配置
        num_envs = args_cli.num_envs
        if args_cli.n_procs > 1:
            args_cli.device = f"cuda:{app_launcher.local_rank}"
            args_cli.device_name = f"cuda:{app_launcher.local_rank}"
            args_cli.multi_gpu = True

        print(f"[Rank {rank}] 创建环境 / Creating environment: {env_name}")
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
            env_name, device=args_cli.device, num_envs=num_envs
        )

        # 关键配置：保留渲染，但准备跳过物理仿真
        env_cfg.sim.render_interval = args_cli.render_interval  # 控制渲染频率
        env_cfg.terminations = None  # 禁用所有终止条件
        if hasattr(env_cfg.events, "reset_camera"):
            del env_cfg.events.reset_camera  # 删除重置相机事件（如果存在）

        # 设置录制器（如果需要）
        if args_cli.output_file is not None:
            from pathlib import Path

            from isaaclab.managers import DatasetExportMode
            from isaaclab_mimic.envs.mdp import ActionStateImageDepthRecorderManagerCfg
            from isaaclab_mimic.utils.datasets import DistDatasetFileHandlerSlave

            output_file = Path(args_cli.output_file)
            if args_cli.record_images:
                env_cfg.recorders = ActionStateImageDepthRecorderManagerCfg()
            else:
                from isaaclab_mimic.envs.mdp import ActionStateRecorderManagerCfg

                env_cfg.recorders = ActionStateRecorderManagerCfg()

            env_cfg.recorders.dataset_file_handler_class_type = (
                DistDatasetFileHandlerSlave
            )
            os.makedirs(output_file.parent, exist_ok=True)
            env_cfg.recorders.dataset_export_mode = (
                DatasetExportMode.EXPORT_SUCCEEDED_ONLY
            )
            env_cfg.recorders.dataset_export_dir_path = str(output_file.parent)
            env_cfg.recorders.dataset_filename = output_file.name

        # 创建环境
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        # 获取模拟上下文
        sim: SimulationContext = env.sim

        # 设置数据队列（如果使用分布式录制）
        if args_cli.output_file is not None and args_cli.n_procs > 1:
            env.set_data_queue(episode_data_queue)

        # 重置环境
        print(f"[Rank {rank}] 重置环境 / Resetting environment...")
        env.reset()

        # 获取分配给此进程的episode索引
        episode_indices_to_replay = [i for i in episode_indices_to_replay_lst[rank]]

        print(
            f"[Rank {rank}] 将处理 {len(episode_indices_to_replay)} 个episodes / Will process {len(episode_indices_to_replay)} episodes"
        )

        # 用于统计的变量
        total_states_set = 0
        replayed_episode_count = 0
        start_time = time.time()

        # 创建默认动作（通常是零动作）
        idle_action = env.cfg.mimic_config.default_actions.repeat(num_envs, 1).to(
            env.device
        )

        # 计算步长时间（用于控制回放速度）
        dt = env.physics_dt / args_cli.replay_speed

        # 使用推理模式处理所有操作
        with torch.inference_mode():
            # 为每个环境创建episode数据映射
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            has_next_action = True

            # 创建进度条
            tqdm_bar = tqdm.tqdm(
                range(len(episode_indices_to_replay)),
                desc=f"[Rank {rank}] 处理episodes",
            )

            while simulation_app.is_running() and not simulation_app.is_exiting():
                # 处理多个环境的动作
                while has_next_action:
                    actions = idle_action
                    has_next_action = False
                    states_to_set = []

                    for env_id in range(num_envs):
                        env_next_action = env_episode_data_map[env_id].get_next_action()

                        if env_next_action is None:
                            # 需要加载新的episode
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
                                    f"[Rank {rank}] {replayed_episode_count:4}: 加载 #{next_episode_index} episode到 env_{env_id}"
                                )
                                episode_data = dataset_file_handler.load_episode(
                                    episode_names[next_episode_index], env.device
                                )
                                env_episode_data_map[env_id] = episode_data

                                # 设置初始状态
                                initial_state = concatenate_state(
                                    [env_episode_data_map[env_id].get_state(2)]
                                )
                                env_episode_data_map[env_id].next_action_index = (
                                    2  # 重置动作索引到1（第一个动作）
                                )
                                initial_state = TensorUtils.to_device(
                                    initial_state, env.device
                                )
                                env_ids_reset = torch.tensor(
                                    [env_id], device=env.device
                                )

                                # 手动调用recorder的pre-reset（如果有recorder）
                                if (
                                    hasattr(env, "recorder_manager")
                                    and env.recorder_manager is not None
                                ):
                                    env.recorder_manager.record_pre_reset(env_ids_reset)

                                # 重置到初始状态
                                env.scene.reset(env_ids_reset)
                                env.scene.reset_to(
                                    initial_state, env_ids_reset, is_relative=True
                                )
                                env.scene.write_data_to_sim()
                                # 检查是否需要渲染（GUI或RTX传感器）
                                is_rendering = sim.has_gui() or sim.has_rtx_sensors()
                                if is_rendering:
                                    # 执行渲染步骤（不运行物理）
                                    sim.render()

                                # 更新场景缓冲区（对相机等传感器很重要）
                                env.scene.update(dt=env.physics_dt)

                                if (
                                    hasattr(env, "observation_manager")
                                    and env.observation_manager is not None
                                ):
                                    env.obs_buf = env.observation_manager.compute()

                                # 手动调用recorder的post-reset（如果有recorder）
                                if (
                                    hasattr(env, "recorder_manager")
                                    and env.recorder_manager is not None
                                ):
                                    env.recorder_manager.record_post_reset(
                                        env_ids_reset
                                    )

                                # 获取第一个动作
                                env_next_action = env_episode_data_map[
                                    env_id
                                ].get_next_action()
                                has_next_action = True
                                print(
                                    f"[Rank {rank}] Env {env_id} 开始episode #{next_episode_index}, 总动作数: {len(episode_data.data['actions'])}"
                                )
                            else:
                                continue
                        else:
                            has_next_action = True

                        actions[env_id] = env_next_action

                        # 获取执行当前动作后的目标状态
                        # env_next_action是刚获取的动作（索引为next_action_index-1）
                        # 执行这个动作后的状态也是同样的索引
                        action_index = (
                            env_episode_data_map[env_id].next_action_index - 1
                        )
                        target_state = env_episode_data_map[env_id].get_state(
                            action_index
                        )
                        states_to_set.append(target_state)

                    if has_next_action:
                        step_start_time = time.time()

                        # 只为有有效动作的环境记录
                        active_env_ids = [
                            i for i, s in enumerate(states_to_set) if s is not None
                        ]

                        # 手动调用recorder的pre-step（如果有recorder）
                        if (
                            hasattr(env, "recorder_manager")
                            and env.recorder_manager is not None
                            and active_env_ids
                        ):
                            # 处理动作数据
                            if (
                                hasattr(env, "action_manager")
                                and env.action_manager is not None
                            ):
                                env.action_manager.process_action(actions)
                            env.recorder_manager.record_pre_step()
                            # 调试信息
                            if replayed_episode_count <= 1:
                                print(
                                    f"[Rank {rank}] Step: 记录pre-step, active_env_ids: {active_env_ids}"
                                )

                        # 连接所有环境的状态
                        valid_states = [s for s in states_to_set if s is not None]
                        if valid_states:
                            concatenated_states = concatenate_state(valid_states)
                            concatenated_states = TensorUtils.to_device(
                                concatenated_states, env.device
                            )

                            # 直接设置状态到场景（不执行物理仿真）
                            env_ids_with_states = torch.tensor(
                                [
                                    i
                                    for i, s in enumerate(states_to_set)
                                    if s is not None
                                ],
                                device=env.device,
                            )
                            env.scene.reset_to(
                                concatenated_states,
                                env_ids_with_states,
                                is_relative=True,
                            )

                            # 将数据写入仿真器（更新可视化）
                            env.scene.write_data_to_sim()

                            # 检查是否需要渲染（GUI或RTX传感器）
                            is_rendering = sim.has_gui() or sim.has_rtx_sensors()
                            if is_rendering:
                                # 执行渲染步骤（不运行物理）
                                sim.render()

                            # 更新场景缓冲区（对相机等传感器很重要）
                            env.scene.update(dt=env.physics_dt)

                            # 手动调用recorder的post-step（如果有recorder）
                            if (
                                hasattr(env, "recorder_manager")
                                and env.recorder_manager is not None
                            ):
                                # 计算观测（recorder可能需要）
                                if (
                                    hasattr(env, "observation_manager")
                                    and env.observation_manager is not None
                                ):
                                    env.obs_buf = env.observation_manager.compute()
                                env.recorder_manager.record_post_step()
                                # 调试信息
                                if replayed_episode_count <= 1:
                                    print(f"[Rank {rank}] Step: 记录post-step")

                            # 更新计数
                            total_states_set += len(valid_states)

                            # 控制回放速度
                            elapsed = time.time() - step_start_time
                            if elapsed < dt:
                                time.sleep(dt - elapsed)

                        # 检查是否有episode完成
                        completed_env_ids = []
                        for env_id in range(num_envs):
                            if env_id in active_env_ids:
                                if (
                                    env_episode_data_map[env_id].get_action(
                                        env_episode_data_map[env_id].next_action_index
                                    )
                                    is None
                                ):
                                    completed_env_ids.append(env_id)

                        if (
                            completed_env_ids
                            and hasattr(env, "recorder_manager")
                            and env.recorder_manager is not None
                        ):
                            completed_env_tensor = torch.tensor(
                                completed_env_ids, device=env.device
                            )
                            # 标记为成功并导出
                            env.recorder_manager.set_success_to_episodes(
                                completed_env_tensor,
                                torch.ones_like(completed_env_tensor).to(
                                    dtype=bool, device=env.device
                                ),
                            )
                            env.recorder_manager.export_episodes(completed_env_tensor)
                            for env_id in completed_env_ids:
                                original_actions = (
                                    env_episode_data_map[env_id].next_action_index - 1
                                )
                                # 重置该环境的episode数据，以便下次循环可以加载新的episode
                                env_episode_data_map[env_id] = EpisodeData()
                            print(
                                f"[Rank {rank}] 导出了 {len(completed_env_ids)} 个完成的episodes"
                            )

                # 检查是否还有未处理的episodes
                if not episode_indices_to_replay and not has_next_action:
                    break

        # 计算并显示统计信息
        elapsed_time = time.time() - start_time
        print(f"\n[Rank {rank}] 处理完成 / Processing completed!")
        print(
            f"[Rank {rank}] 总计处理状态数 / Total states processed: {total_states_set}"
        )
        print(
            f"[Rank {rank}] 处理episode数 / Episodes processed: {replayed_episode_count}"
        )
        print(f"[Rank {rank}] 耗时 / Time elapsed: {elapsed_time:.2f} 秒")
        print(
            f"[Rank {rank}] 平均处理速度 / Average processing speed: {total_states_set/elapsed_time:.2f} 状态/秒"
        )

        # 等待数据保存完成
        if args_cli.output_file is not None and args_cli.n_procs > 1:
            stop_queue.get()
            episode_data_queue.join()

        # 关闭环境和数据集
        dataset_file_handler.close()
        env.close()

    except Exception as e:
        print(f"[Rank {rank}] 错误 / Error: {e}")
        import traceback

        traceback.print_exc()


def main_single_process():
    """单进程主函数 / Single process main function"""

    # 启动模拟器（带渲染模式）
    print("启动Isaac Sim（渲染模式）/ Launching Isaac Sim (with rendering)...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 导入必要的模块
    import isaaclab_mimic.envs  # noqa: F401
    import isaaclab_mimic.tasks  # noqa: F401
    import isaaclab_mimic.utils.tensor_utils as TensorUtils
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.sim import SimulationContext
    from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # 加载输入数据集
    print(f"\n加载数据集 / Loading dataset: {args_cli.input_file}")
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(
            f"输入数据集文件不存在 / Input dataset file not found: {args_cli.input_file}"
        )

    dataset_file_handler = ZarrCompatibleDatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)

    # 获取数据集信息
    env_name = args_cli.task or dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()
    episode_names = list(dataset_file_handler.get_episode_names())

    print(
        f"数据集包含 {episode_count} 个episodes / Dataset contains {episode_count} episodes"
    )

    if episode_count == 0:
        print("数据集中没有找到episodes / No episodes found in dataset")
        simulation_app.close()
        return

    # 解析环境配置
    print(f"\n创建环境 / Creating environment: {env_name}")
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        env_name, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # 关键配置：保留渲染，但准备跳过物理仿真
    env_cfg.sim.render_interval = args_cli.render_interval  # 控制渲染频率
    env_cfg.terminations = None  # 禁用所有终止条件

    # 设置录制器（如果需要）
    if args_cli.output_file is not None:
        from pathlib import Path

        from isaaclab.managers import DatasetExportMode
        from isaaclab_mimic.envs.mdp import ActionStateImageDepthRecorderManagerCfg
        from isaaclab_mimic.utils.datasets import (
            ZarrCompatibleDatasetFileHandler as OutputHandler,
        )

        output_file = Path(args_cli.output_file)
        if args_cli.record_images:
            env_cfg.recorders = ActionStateImageDepthRecorderManagerCfg()
        else:
            from isaaclab_mimic.envs.mdp import ActionStateRecorderManagerCfg

            env_cfg.recorders = ActionStateRecorderManagerCfg()

        env_cfg.recorders.dataset_file_handler_class_type = OutputHandler
        os.makedirs(output_file.parent, exist_ok=True)
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        env_cfg.recorders.dataset_export_dir_path = str(output_file.parent)
        env_cfg.recorders.dataset_filename = output_file.name

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 获取模拟上下文
    sim: SimulationContext = env.sim

    # 重置环境
    print("重置环境 / Resetting environment...")
    env.reset()

    # 确定要处理的episode索引
    if args_cli.episode_indices:
        episodes_to_process = [i for i in args_cli.episode_indices if i < episode_count]
    else:
        episodes_to_process = list(range(episode_count))

    print(
        f"\n将处理 {len(episodes_to_process)} 个episodes / Will process {len(episodes_to_process)} episodes"
    )

    # 用于统计的变量
    total_states_set = 0
    total_mismatches = 0
    start_time = time.time()

    # 创建默认动作（通常是零动作）
    idle_action = env.cfg.mimic_config.default_actions.repeat(args_cli.num_envs, 1).to(
        env.device
    )

    # 计算步长时间（用于控制回放速度）
    dt = env.physics_dt / args_cli.replay_speed

    # 用于跟踪每个环境的统计信息
    env_stats = {
        env_id: {"actions_processed": 0, "states_set": 0}
        for env_id in range(args_cli.num_envs)
    }

    # 使用推理模式处理所有操作
    with torch.inference_mode():
        from isaaclab.utils.datasets import EpisodeData

        # 为每个环境创建episode数据映射
        env_episode_data_map = {
            index: EpisodeData() for index in range(args_cli.num_envs)
        }
        replayed_episode_count = 0
        episode_indices_to_replay = episodes_to_process.copy()

        # 创建进度条
        tqdm_bar = tqdm.tqdm(
            range(len(episodes_to_process)), desc="处理episodes / Processing episodes"
        )

        has_next_action = True
        while (has_next_action and episode_indices_to_replay) or any(
            not env_episode_data_map[i].is_empty() for i in range(args_cli.num_envs)
        ):
            actions = idle_action.clone()
            has_next_action = False
            states_to_set = []
            env_actions_info = {}  # 记录每个环境的动作信息

            for env_id in range(args_cli.num_envs):
                # 先检查当前环境是否还有动作
                if not env_episode_data_map[env_id].is_empty():
                    # 检查是否到达episode末尾
                    current_episode_actions = env_episode_data_map[env_id].data.get(
                        "actions", []
                    )
                    if env_episode_data_map[env_id].next_action_index < len(
                        current_episode_actions
                    ):
                        # 还有动作可以获取
                        env_next_action = env_episode_data_map[env_id].get_next_action()
                        has_next_action = True

                        actions[env_id] = env_next_action

                        # 获取执行当前动作后的目标状态
                        action_index = (
                            env_episode_data_map[env_id].next_action_index - 1
                        )
                        target_state = env_episode_data_map[env_id].get_state(
                            action_index
                        )
                        states_to_set.append(target_state)
                        env_actions_info[env_id] = {
                            "has_action": True,
                            "action_index": action_index,
                            "total_actions": len(current_episode_actions),
                        }
                        env_stats[env_id]["actions_processed"] += 1
                    else:
                        # Episode已完成，需要加载新的
                        states_to_set.append(None)
                        env_actions_info[env_id] = {
                            "has_action": False,
                            "completed": True,
                        }
                else:
                    # 需要加载新的episode
                    if episode_indices_to_replay:
                        next_episode_index = episode_indices_to_replay.pop(0)
                        tqdm_bar.update(1)

                        replayed_episode_count += 1
                        print(
                            f"{replayed_episode_count:4}: 加载 #{next_episode_index} episode到 env_{env_id}"
                        )
                        episode_data = dataset_file_handler.load_episode(
                            episode_names[next_episode_index], env.device
                        )
                        env_episode_data_map[env_id] = episode_data

                        # 设置初始状态
                        initial_state = episode_data.get_initial_state()
                        initial_state = concatenate_state(
                            [env_episode_data_map[env_id].get_state(2)]
                        )
                        env_episode_data_map[env_id].next_action_index = (
                            2  # 重置动作索引到1（第一个动作）
                        )
                        # 获取第一个状态
                        initial_state = TensorUtils.to_device(initial_state, env.device)
                        env_ids_reset = torch.tensor([env_id], device=env.device)

                        # 手动调用recorder的pre-reset（如果有recorder）
                        if (
                            hasattr(env, "recorder_manager")
                            and env.recorder_manager is not None
                        ):
                            env.recorder_manager.record_pre_reset(env_ids_reset)

                        # 重置到初始状态
                        env.scene.reset(env_ids_reset)
                        env.scene.reset_to(
                            initial_state, env_ids_reset, is_relative=True
                        )
                        env.scene.write_data_to_sim()
                        # 检查是否需要渲染（GUI或RTX传感器）
                        is_rendering = sim.has_gui() or sim.has_rtx_sensors()
                        if is_rendering:
                            # 执行渲染步骤（不运行物理）
                            sim.render()

                        # 更新场景缓冲区（对相机等传感器很重要）
                        env.scene.update(dt=env.physics_dt)

                        # 手动调用recorder的post-reset（如果有recorder）
                        if (
                            hasattr(env, "recorder_manager")
                            and env.recorder_manager is not None
                        ):
                            env.recorder_manager.record_post_reset(env_ids_reset)

                        if (
                            hasattr(env, "observation_manager")
                            and env.observation_manager is not None
                        ):
                            env.obs_buf = env.observation_manager.compute()

                        # 获取第一个动作
                        env_next_action = env_episode_data_map[env_id].get_next_action()
                        if env_next_action is not None:
                            has_next_action = True
                            actions[env_id] = env_next_action

                            # 获取执行当前动作后的目标状态
                            action_index = (
                                env_episode_data_map[env_id].next_action_index - 1
                            )
                            target_state = env_episode_data_map[env_id].get_state(
                                action_index
                            )
                            states_to_set.append(target_state)
                            env_actions_info[env_id] = {
                                "has_action": True,
                                "action_index": action_index,
                                "total_actions": len(episode_data.data["actions"]),
                            }
                            env_stats[env_id]["actions_processed"] += 1
                            print(
                                f"Env {env_id} 开始episode #{next_episode_index}, 总动作数: {len(episode_data.data['actions'])}"
                            )
                        else:
                            states_to_set.append(None)
                            env_actions_info[env_id] = {"has_action": False}
                    else:
                        states_to_set.append(None)
                        env_actions_info[env_id] = {"has_action": False}

            if has_next_action or any(
                info.get("has_action", False) for info in env_actions_info.values()
            ):
                step_start_time = time.time()

                # 只为有有效动作的环境记录
                active_env_ids = [
                    i
                    for i, info in env_actions_info.items()
                    if info.get("has_action", False)
                ]

                # 手动调用recorder的pre-step（如果有recorder）
                if (
                    hasattr(env, "recorder_manager")
                    and env.recorder_manager is not None
                    and active_env_ids
                ):
                    # 处理动作数据
                    if (
                        hasattr(env, "action_manager")
                        and env.action_manager is not None
                    ):
                        env.action_manager.process_action(actions)
                    env.recorder_manager.record_pre_step()

                # 连接所有环境的状态
                valid_states = [
                    s
                    for i, s in enumerate(states_to_set)
                    if s is not None and i in active_env_ids
                ]
                if valid_states:
                    concatenated_states = concatenate_state(valid_states)
                    concatenated_states = TensorUtils.to_device(
                        concatenated_states, env.device
                    )

                    # 直接设置状态到场景（不执行物理仿真）
                    env_ids_with_states = torch.tensor(
                        active_env_ids, device=env.device
                    )
                    env.scene.reset_to(
                        concatenated_states, env_ids_with_states, is_relative=True
                    )

                    # 将数据写入仿真器（更新可视化）
                    env.scene.write_data_to_sim()

                    # 检查是否需要渲染（GUI或RTX传感器）
                    is_rendering = sim.has_gui() or sim.has_rtx_sensors()
                    if is_rendering:
                        # 执行渲染步骤（不运行物理）
                        sim.render()

                    # 更新场景缓冲区（对相机等传感器很重要）
                    env.scene.update(dt=env.physics_dt)

                    # 手动调用recorder的post-step（如果有recorder）
                    if (
                        hasattr(env, "recorder_manager")
                        and env.recorder_manager is not None
                    ):
                        # 计算观测（recorder可能需要）
                        if (
                            hasattr(env, "observation_manager")
                            and env.observation_manager is not None
                        ):
                            env.obs_buf = env.observation_manager.compute()
                        env.recorder_manager.record_post_step()

                    # 更新计数
                    total_states_set += len(valid_states)
                    for env_id in active_env_ids:
                        env_stats[env_id]["states_set"] += 1

                # 检查是否有episode完成
                completed_env_ids = []
                for env_id, info in env_actions_info.items():
                    if info.get("has_action", False):
                        # 检查是否是最后一个动作
                        action_idx = info["action_index"]
                        total_actions = info["total_actions"]
                        if action_idx == total_actions - 1:  # 最后一个动作
                            completed_env_ids.append(env_id)
                            print(
                                f"Env {env_id} 完成episode, 处理了 {action_idx + 1}/{total_actions} 个动作"
                            )

                if (
                    completed_env_ids
                    and hasattr(env, "recorder_manager")
                    and env.recorder_manager is not None
                ):
                    completed_env_tensor = torch.tensor(
                        completed_env_ids, device=env.device
                    )
                    # 标记为成功并导出
                    env.recorder_manager.set_success_to_episodes(
                        completed_env_tensor,
                        torch.ones_like(completed_env_tensor).to(
                            dtype=bool, device=env.device
                        ),
                    )
                    env.recorder_manager.export_episodes(completed_env_tensor)
                    for env_id in completed_env_ids:
                        # 重置该环境的episode数据，以便下次循环可以加载新的episode
                        env_episode_data_map[env_id] = EpisodeData()
                    print(f"导出了 {len(completed_env_ids)} 个完成的episodes")

                # 更新进度条信息
                tqdm_bar.set_postfix(
                    {
                        "总状态数/Total states": total_states_set,
                        "已处理episodes/Episodes": replayed_episode_count,
                    }
                )

    # 打印每个环境的统计信息
    print("\n每个环境的处理统计 / Per-environment processing statistics:")
    for env_id, stats in env_stats.items():
        print(
            f"Env {env_id}: 处理动作数={stats['actions_processed']}, 设置状态数={stats['states_set']}"
        )

    # 计算并显示统计信息
    elapsed_time = time.time() - start_time
    print(f"\n处理完成 / Processing completed!")
    print(f"总计处理状态数 / Total states processed: {total_states_set}")
    print(f"总计状态不匹配数 / Total state mismatches: {total_mismatches}")
    print(f"耗时 / Time elapsed: {elapsed_time:.2f} 秒")
    print(
        f"平均处理速度 / Average processing speed: {total_states_set/elapsed_time:.2f} 状态/秒"
    )

    # 关闭环境和数据集
    dataset_file_handler.close()
    env.close()

    # 关闭模拟应用
    print("\n关闭模拟器 / Closing simulator...")
    simulation_app.close()


if __name__ == "__main__":
    if args_cli.distributed and args_cli.n_procs > 1:
        # 多进程模式
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29523"
        mp.set_start_method("spawn")

        # 预先加载数据集以获取episode数量
        from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler

        dataset_file_handler = ZarrCompatibleDatasetFileHandler()
        dataset_file_handler.open(args_cli.input_file)
        episode_count = dataset_file_handler.get_num_episodes()
        dataset_file_handler.close()

        # 确定要处理的episode索引
        if args_cli.episode_indices:
            episodes_to_process = [
                i for i in args_cli.episode_indices if i < episode_count
            ]
        else:
            episodes_to_process = list(range(episode_count))

        # 分配episodes到各个进程
        world_size = args_cli.n_procs
        episode_indices_to_replay_lst = [[] for i in range(world_size)]

        for i, episode_idx in enumerate(episodes_to_process):
            episode_indices_to_replay_lst[i % world_size].append(episode_idx)

        print(
            f"启动 {world_size} 个进程进行并行处理 / Launching {world_size} processes for parallel processing"
        )

        if args_cli.output_file is not None:
            # 创建共享队列
            shared_queue = JoinableQueue()
            stop_queue = mp.Queue()

            # 获取环境名称
            dataset_file_handler = ZarrCompatibleDatasetFileHandler()
            dataset_file_handler.open(args_cli.input_file)
            env_name = args_cli.task or dataset_file_handler.get_env_name()
            dataset_file_handler.close()

            # 启动工作进程
            mp.spawn(
                replay_worker,
                args=(
                    world_size,
                    shared_queue,
                    episode_indices_to_replay_lst,
                    stop_queue,
                ),
                nprocs=world_size,
                join=False,
                daemon=True,
            )

            # 启动数据保存进程
            output_file = Path(args_cli.output_file)
            dataset_export_dir_path = str(output_file.parent)
            dataset_filename = output_file.name

            try:
                mp.spawn(
                    save_episode_data,
                    args=(
                        shared_queue,
                        len(episodes_to_process),
                        dataset_export_dir_path,
                        dataset_filename,
                        env_name,
                        stop_queue,
                    ),
                    join=True,
                    daemon=True,
                )
            except KeyboardInterrupt:
                print(
                    "\n程序被用户中断。退出... / Program interrupted by user. Exiting..."
                )
        else:
            # 不需要录制，只需要回放
            mp.spawn(
                replay_worker,
                args=(world_size, None, episode_indices_to_replay_lst, None),
                nprocs=world_size,
                join=True,
            )

    else:
        # 单进程模式
        print("单进程模式 / Single process mode")
        main_single_process()
