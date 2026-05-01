# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch


def compare_states(state_from_dataset, runtime_state, device, num_envs):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = torch.tensor([True] * num_envs, device=device)
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][
                    state_name
                ]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(
                        f"State shape of {state_name} for asset {asset_name} don't match"
                    )
                states_matched = torch.logical_and(
                    states_matched,
                    (
                        abs(dataset_asset_state.to(device) - runtime_asset_state) < 0.01
                    ).all(dim=-1),
                )
    return states_matched


def concatenate_state(state_lst):
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
