# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
脚本功能：从数据集中读取状态并设置到环境中，不进行渲染
主要用途：快速验证状态设置、数据处理、批量状态分析等场景

Script to set states from dataset without rendering.
Main use: Quick state validation, data processing, batch state analysis.
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import torch
import tqdm
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(
    description="Set states from dataset without rendering in Isaac Lab environments."
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
    "--disable_fabric",
    action="store_true",
    help="禁用fabric以加速状态设置 / Disable fabric for faster state setting",
)
parser.add_argument(
    "--skip_physics",
    action="store_true",
    help="跳过物理仿真步骤 / Skip physics simulation steps",
)

# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 设置无头模式，禁用渲染
args_cli.headless = True


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


def main():
    """主函数 / Main function"""

    # 启动模拟器（无渲染模式）
    print("启动Isaac Sim（无渲染模式）/ Launching Isaac Sim (no rendering mode)...")
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
        return

    # 解析环境配置
    print(f"\n创建环境 / Creating environment: {env_name}")
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        env_name, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # 关键配置：禁用渲染相关功能
    env_cfg.sim.render_interval = 0  # 设置渲染间隔为0，禁用自动渲染
    env_cfg.viewer = None  # 禁用viewer
    env_cfg.terminations = None  # 禁用所有终止条件

    # 可选：禁用fabric以提高性能
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 获取模拟上下文并设置渲染模式
    sim: SimulationContext = env.sim
    sim.set_render_mode(SimulationContext.RenderMode.NO_GUI_OR_RENDERING)

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
    idle_action = env.cfg.mimic_config.default_actions.repeat(args_cli.num_envs, 1)

    # 使用推理模式处理所有操作
    with torch.inference_mode():
        # 创建进度条
        pbar = tqdm.tqdm(episodes_to_process, desc="处理episodes / Processing episodes")

        for episode_idx in pbar:
            # 加载episode数据
            episode_data = dataset_file_handler.load_episode(
                episode_names[episode_idx], env.device
            )

            # 获取初始状态并设置
            initial_state = episode_data.get_initial_state()
            env_ids = torch.tensor([0], device=env.device)  # 使用第一个环境

            # 设置初始状态
            env.reset_to(initial_state, env_ids, is_relative=True)

            # 遍历episode中的所有状态
            num_steps = len(episode_data)
            state_mismatches_in_episode = 0

            for step_idx in range(num_steps):
                # 获取当前步的动作和状态
                action = episode_data.get_action(step_idx)
                target_state = episode_data.get_state(step_idx)

                # 准备动作张量
                actions = idle_action.clone()
                actions[0] = action

                if not args_cli.skip_physics:
                    # 执行动作（不渲染）
                    env.step(actions)

                    # 获取当前运行时状态
                    current_state = env.scene.get_state(is_relative=True)

                    # 比较状态
                    target_state_batched = TensorUtils.map_tensor(
                        target_state, lambda x: x.unsqueeze(0) if x.dim() == 1 else x
                    )
                    states_matched = compare_states(
                        target_state_batched,
                        current_state,
                        device=env.device,
                        num_envs=1,
                    )

                    # 如果状态不匹配，重置到目标状态
                    if not states_matched[0]:
                        state_mismatches_in_episode += 1
                        env.scene.reset_to(
                            target_state_batched, env_ids, is_relative=True
                        )
                else:
                    # 直接设置状态，跳过物理仿真
                    target_state_batched = TensorUtils.map_tensor(
                        target_state, lambda x: x.unsqueeze(0) if x.dim() == 1 else x
                    )
                    env.scene.reset_to(target_state_batched, env_ids, is_relative=True)

                # 将数据写入仿真器（但不渲染）
                env.scene.write_data_to_sim()

                total_states_set += 1

            total_mismatches += state_mismatches_in_episode

            # 更新进度条信息
            pbar.set_postfix(
                {
                    "总状态数/Total states": total_states_set,
                    "不匹配数/Mismatches": total_mismatches,
                    "本episode不匹配/Episode mismatches": state_mismatches_in_episode,
                }
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
    # 运行主函数
    main()
