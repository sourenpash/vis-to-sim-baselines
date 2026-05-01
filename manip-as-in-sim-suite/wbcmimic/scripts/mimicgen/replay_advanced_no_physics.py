# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
高级replay脚本：支持多种物理仿真控制模式
Advanced replay script: Support multiple physics simulation control modes

模式说明 / Mode descriptions:
1. no_physics: 完全跳过物理仿真，只设置状态和渲染
2. physics_no_step: 运行物理引擎但不步进（用于碰撞检测等）
3. physics_minimal: 最小化物理步进（每N帧步进一次）
4. physics_full: 完整物理仿真（默认行为）
"""

import argparse
import os
import time
from enum import Enum
from pathlib import Path

import gymnasium as gym
import torch
import tqdm
from isaaclab.app import AppLauncher


class PhysicsMode(Enum):
    NO_PHYSICS = "no_physics"
    PHYSICS_NO_STEP = "physics_no_step"
    PHYSICS_MINIMAL = "physics_minimal"
    PHYSICS_FULL = "physics_full"


# 添加命令行参数
parser = argparse.ArgumentParser(
    description="Advanced replay with flexible physics control."
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
    help="要处理的episode索引列表 / Episode indices to process",
)
parser.add_argument(
    "--physics_mode",
    type=str,
    default="no_physics",
    choices=[mode.value for mode in PhysicsMode],
    help="物理仿真模式 / Physics simulation mode",
)
parser.add_argument(
    "--physics_step_interval",
    type=int,
    default=10,
    help="物理步进间隔（仅在physics_minimal模式下使用）/ Physics step interval (only for physics_minimal mode)",
)
parser.add_argument(
    "--render_interval", type=int, default=1, help="渲染间隔 / Render interval"
)
parser.add_argument(
    "--replay_speed",
    type=float,
    default=1.0,
    help="回放速度倍率 / Replay speed multiplier",
)
parser.add_argument(
    "--compare_states",
    action="store_true",
    help="是否比较设置的状态和实际状态 / Whether to compare set states with actual states",
)
parser.add_argument(
    "--visualize_contacts",
    action="store_true",
    help="可视化接触点（需要physics_no_step或更高模式）/ Visualize contact points",
)
parser.add_argument(
    "--camera_follow", action="store_true", help="相机跟随机器人 / Camera follows robot"
)

# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 设置渲染模式
args_cli.headless = False


def compare_states(state_from_dataset, runtime_state, device, num_envs):
    """比较状态的辅助函数"""
    states_matched = torch.tensor([True] * num_envs, device=device)

    for asset_type in ["articulation", "rigid_object"]:
        if asset_type not in runtime_state:
            continue

        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][
                    state_name
                ]

                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(
                        f"State shape mismatch for {state_name} of {asset_name}"
                    )

                states_matched = torch.logical_and(
                    states_matched,
                    (
                        abs(dataset_asset_state.to(device) - runtime_asset_state) < 0.01
                    ).all(dim=1),
                )

    return states_matched


def main():
    """主函数"""

    # 启动模拟器
    print(
        f"启动Isaac Sim - 物理模式: {args_cli.physics_mode} / Launching Isaac Sim - Physics mode: {args_cli.physics_mode}"
    )
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 导入必要的模块
    import bytemini_sim.tasks  # noqa: F401
    import isaaclab_mimic.envs  # noqa: F401
    import isaaclab_mimic.utils.tensor_utils as TensorUtils
    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.sim import SimulationContext
    from isaaclab_mimic.utils.datasets import ZarrCompatibleDatasetFileHandler
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # 加载数据集
    print(f"\n加载数据集 / Loading dataset: {args_cli.input_file}")
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"Dataset file not found: {args_cli.input_file}")

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
        print("No episodes found in dataset")
        return

    # 解析环境配置
    print(f"\n创建环境 / Creating environment: {env_name}")
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        env_name, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # 配置渲染
    env_cfg.sim.render_interval = args_cli.render_interval
    env_cfg.terminations = None  # 禁用终止条件

    # 根据物理模式调整配置
    physics_mode = PhysicsMode(args_cli.physics_mode)
    if physics_mode == PhysicsMode.NO_PHYSICS:
        # 尝试禁用物理相关的更新
        env_cfg.sim.physics_prim_path = None
        print("物理模式：完全禁用 / Physics mode: Completely disabled")
    elif physics_mode == PhysicsMode.PHYSICS_NO_STEP:
        print("物理模式：启用但不步进 / Physics mode: Enabled but no stepping")
    elif physics_mode == PhysicsMode.PHYSICS_MINIMAL:
        print(
            f"物理模式：最小化（每{args_cli.physics_step_interval}步） / Physics mode: Minimal (every {args_cli.physics_step_interval} steps)"
        )
    else:
        print("物理模式：完整仿真 / Physics mode: Full simulation")

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    sim: SimulationContext = env.sim

    # 重置环境
    print("重置环境 / Resetting environment...")
    env.reset()

    # 确定要处理的episodes
    if args_cli.episode_indices:
        episodes_to_process = [i for i in args_cli.episode_indices if i < episode_count]
    else:
        episodes_to_process = list(range(episode_count))

    print(
        f"\n将处理 {len(episodes_to_process)} 个episodes / Will process {len(episodes_to_process)} episodes"
    )

    # 统计变量
    total_states_set = 0
    total_mismatches = 0
    start_time = time.time()
    physics_steps_taken = 0

    # 创建默认动作
    idle_action = env.cfg.mimic_config.default_actions.repeat(args_cli.num_envs, 1)

    # 计算时间步长
    dt = env.physics_dt / args_cli.replay_speed

    # 主循环
    with torch.inference_mode():
        pbar = tqdm.tqdm(episodes_to_process, desc="处理episodes / Processing episodes")

        for episode_idx in pbar:
            # 加载episode
            episode_data = dataset_file_handler.load_episode(
                episode_names[episode_idx], env.device
            )

            # 设置初始状态
            initial_state = episode_data.get_initial_state()
            env_ids = torch.tensor([0], device=env.device)
            env.reset_to(initial_state, env_ids, is_relative=True)

            # 遍历所有步骤
            num_steps = len(episode_data)
            state_mismatches_in_episode = 0

            for step_idx in range(num_steps):
                step_start_time = time.time()

                # 获取目标状态
                target_state = episode_data.get_state(step_idx)
                target_state_batched = TensorUtils.map_tensor(
                    target_state, lambda x: x.unsqueeze(0) if x.dim() == 1 else x
                )

                # 设置状态
                env.scene.reset_to(target_state_batched, env_ids, is_relative=True)

                # 根据物理模式决定是否步进
                if physics_mode == PhysicsMode.PHYSICS_FULL:
                    # 完整物理仿真
                    action = episode_data.get_action(step_idx)
                    actions = idle_action.clone()
                    actions[0] = action
                    env.step(actions)
                    physics_steps_taken += 1
                elif (
                    physics_mode == PhysicsMode.PHYSICS_MINIMAL
                    and step_idx % args_cli.physics_step_interval == 0
                ):
                    # 最小化物理步进
                    env.scene.write_data_to_sim()
                    sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                    physics_steps_taken += 1
                elif physics_mode == PhysicsMode.PHYSICS_NO_STEP:
                    # 只更新场景，不步进物理
                    env.scene.write_data_to_sim()
                    env.scene.update(dt=0)  # 零时间步更新
                else:
                    # NO_PHYSICS模式：只写入数据
                    env.scene.write_data_to_sim()

                # 渲染
                sim.render()

                # 如果需要，比较状态
                if args_cli.compare_states and physics_mode != PhysicsMode.NO_PHYSICS:
                    current_state = env.scene.get_state(is_relative=True)
                    states_matched = compare_states(
                        target_state_batched,
                        current_state,
                        device=env.device,
                        num_envs=1,
                    )
                    if not states_matched[0]:
                        state_mismatches_in_episode += 1

                # 如果需要，更新相机位置
                if args_cli.camera_follow:
                    # 获取机器人位置
                    robot_pos = env.scene["robot"].data.root_pos_w[0]
                    # 设置相机跟随
                    camera_pos = robot_pos + torch.tensor(
                        [2.0, 2.0, 2.0], device=env.device
                    )
                    sim.set_camera_view(
                        camera_pos.cpu().numpy(), robot_pos.cpu().numpy()
                    )

                total_states_set += 1

                # 控制回放速度
                elapsed = time.time() - step_start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            total_mismatches += state_mismatches_in_episode

            # 更新进度条
            pbar.set_postfix(
                {
                    "状态数/States": total_states_set,
                    "物理步数/Physics steps": physics_steps_taken,
                    "不匹配/Mismatches": (
                        state_mismatches_in_episode
                        if args_cli.compare_states
                        else "N/A"
                    ),
                }
            )

    # 显示统计信息
    elapsed_time = time.time() - start_time
    print(f"\n===== 回放完成 / Replay Completed =====")
    print(f"物理模式 / Physics mode: {args_cli.physics_mode}")
    print(f"总状态数 / Total states: {total_states_set}")
    print(f"物理步数 / Physics steps: {physics_steps_taken}")
    print(
        f"物理步比例 / Physics step ratio: {physics_steps_taken/total_states_set:.2%}"
    )
    if args_cli.compare_states:
        print(f"状态不匹配 / State mismatches: {total_mismatches}")
    print(f"总耗时 / Total time: {elapsed_time:.2f} 秒")
    print(f"平均FPS / Average FPS: {total_states_set/elapsed_time:.2f}")

    # 清理
    dataset_file_handler.close()
    env.close()

    print("\n关闭模拟器 / Closing simulator...")
    simulation_app.close()


if __name__ == "__main__":
    main()
