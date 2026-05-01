#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Script to downsample episodes in a zarr dataset.
Reduces the number of timesteps per episode by a given factor.
"""

import argparse
import math
import os
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from tqdm import tqdm


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if (
            this_chunk_bytes <= target_chunk_bytes
            and next_chunk_bytes > target_chunk_bytes
        ):
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(
        this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


def resolve_compressor(compressor="default"):
    if compressor == "default":
        compressor = numcodecs.Blosc(
            cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
        )
    elif compressor == "disk":
        compressor = numcodecs.Blosc(
            "zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
        )
    return compressor


def downsample_array(arr, factor, axis=0):
    """Downsample an array along a given axis by selecting every 'factor' elements."""
    indices = np.arange(0, arr.shape[axis], factor)
    return np.take(arr, indices, axis=axis)


def downsample_nested_dict(data_dict, factor, episode_start, episode_end):
    """Recursively downsample nested dictionary of arrays."""
    result = {}

    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            result[key] = downsample_nested_dict(
                value, factor, episode_start, episode_end
            )
        elif isinstance(value, (np.ndarray, zarr.Array)):
            # Handle arrays - only downsample if first dimension matches episode length
            if len(value.shape) > 0 and value.shape[0] == (episode_end - episode_start):
                # This is episode data that needs downsampling
                episode_data = value[episode_start:episode_end]
                downsampled = downsample_array(episode_data, factor, axis=0)
                result[key] = downsampled
            else:
                # This might be metadata or different structure, keep as is
                result[key] = value[:]
        else:
            # Keep other types as is
            result[key] = value

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Downsample episodes in a zarr dataset"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input zarr file path"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output zarr file path"
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=2,
        help="Downsample factor (default: 2)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process",
    )
    args = parser.parse_args()

    # Open input zarr file
    print(f"Opening input file: {args.input_file}")
    store_in = zarr.open(args.input_file, mode="r")

    # Create output zarr file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating output file: {args.output_file}")
    store_out = zarr.open(args.output_file, mode="w")

    # Get episode information
    episode_ends = store_in["meta/episode_ends"][:]
    num_episodes = len(episode_ends)

    if args.max_episodes:
        num_episodes = min(num_episodes, args.max_episodes)

    print(
        f"Processing {num_episodes} episodes with downsample factor {args.downsample_factor}"
    )

    # Process metadata first (copy without modification except episode_ends)
    print("Copying metadata...")
    meta_group = store_out.create_group("meta")
    for key in store_in["meta"].keys():
        if key != "episode_ends":
            meta_item = store_in[f"meta/{key}"]
            if isinstance(meta_item, zarr.Group):
                # Handle nested groups recursively
                def copy_group(src_group, dst_parent, group_name):
                    dst_group = dst_parent.create_group(group_name)
                    for sub_key in src_group.keys():
                        sub_item = src_group[sub_key]
                        if isinstance(sub_item, zarr.Group):
                            copy_group(sub_item, dst_group, sub_key)
                        else:
                            sub_data = sub_item[:]
                            chunk_size = get_optimal_chunks(
                                shape=sub_data.shape, dtype=sub_data.dtype
                            )
                            dst_group.create_dataset(
                                sub_key,
                                data=sub_data,
                                chunks=chunk_size,
                                compressor=resolve_compressor("default"),
                            )

                copy_group(meta_item, meta_group, key)
            else:
                # Handle arrays normally
                meta_data = meta_item[:]
                if len(meta_data.shape) > 0:  # Only apply chunking to non-scalar data
                    chunk_size = get_optimal_chunks(
                        shape=meta_data.shape, dtype=meta_data.dtype
                    )
                    meta_group.create_dataset(
                        key,
                        data=meta_data,
                        chunks=chunk_size,
                        compressor=resolve_compressor("default"),
                    )
                else:
                    meta_group[key] = meta_data

    # Create data group
    data_group = store_out.create_group("data")

    # Initialize storage for downsampled data
    all_data = {
        "actions": [],
        "obs": {},
        "states": {},
        "wbc_step": [],
        "wbc_target": {},
    }

    # Process initial_state separately (it's per-episode, not per-timestep)
    if "initial_state" in store_in["data"]:
        print("Processing initial states...")
        # Copy initial state structure - it's organized by category/object/state_type
        initial_state_group = data_group.create_group("initial_state")
        for category in ["articulation", "rigid_object"]:
            if category in store_in["data/initial_state"]:
                cat_group = initial_state_group.create_group(category)
                for obj_name in store_in[f"data/initial_state/{category}"]:
                    obj_group = cat_group.create_group(obj_name)
                    for state_type in store_in[
                        f"data/initial_state/{category}/{obj_name}"
                    ]:
                        # Initial states are per-episode, so we only take the first num_episodes
                        initial_data = store_in[
                            f"data/initial_state/{category}/{obj_name}/{state_type}"
                        ][:num_episodes]
                        chunk_size = get_optimal_chunks(
                            shape=initial_data.shape, dtype=initial_data.dtype
                        )
                        obj_group.create_dataset(
                            state_type,
                            data=initial_data,
                            chunks=chunk_size,
                            compressor=resolve_compressor("default"),
                        )

    # Initialize nested dictionaries for obs and states
    for key in store_in["data/obs"].keys():
        if isinstance(store_in[f"data/obs/{key}"], zarr.Group):
            all_data["obs"][key] = {}
        else:
            all_data["obs"][key] = []

    for category in ["articulation", "rigid_object"]:
        if category in store_in["data/states"]:
            all_data["states"][category] = {}
            for obj_name in store_in[f"data/states/{category}"]:
                all_data["states"][category][obj_name] = {}
                for state_type in store_in[f"data/states/{category}/{obj_name}"]:
                    all_data["states"][category][obj_name][state_type] = []

    if "wbc_target" in store_in["data"]:
        for key in store_in["data/wbc_target"].keys():
            all_data["wbc_target"][key] = []

    # Process each episode
    new_episode_ends = []
    current_end = 0

    for ep_idx in tqdm(range(num_episodes), desc="Processing episodes"):
        # Get episode boundaries
        ep_start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        ep_end = episode_ends[ep_idx]
        ep_length = ep_end - ep_start

        # Calculate downsampled length
        downsampled_length = len(range(0, ep_length, args.downsample_factor))

        # Process actions
        episode_actions = store_in["data/actions"][ep_start:ep_end]
        downsampled_actions = downsample_array(episode_actions, args.downsample_factor)
        all_data["actions"].append(downsampled_actions)

        # Process observations
        for obs_key in store_in["data/obs"].keys():
            if isinstance(store_in[f"data/obs/{obs_key}"], zarr.Group):
                # Handle nested observations (like depths)
                for sub_key in store_in[f"data/obs/{obs_key}"].keys():
                    if sub_key not in all_data["obs"][obs_key]:
                        all_data["obs"][obs_key][sub_key] = []
                    episode_data = store_in[f"data/obs/{obs_key}/{sub_key}"][
                        ep_start:ep_end
                    ]
                    downsampled_data = downsample_array(
                        episode_data, args.downsample_factor
                    )
                    all_data["obs"][obs_key][sub_key].append(downsampled_data)
            else:
                # Handle flat observations
                episode_data = store_in[f"data/obs/{obs_key}"][ep_start:ep_end]
                downsampled_data = downsample_array(
                    episode_data, args.downsample_factor
                )
                all_data["obs"][obs_key].append(downsampled_data)

        # Process states
        for category in ["articulation", "rigid_object"]:
            if category in store_in["data/states"]:
                for obj_name in store_in[f"data/states/{category}"]:
                    for state_type in store_in[f"data/states/{category}/{obj_name}"]:
                        episode_data = store_in[
                            f"data/states/{category}/{obj_name}/{state_type}"
                        ][ep_start:ep_end]
                        downsampled_data = downsample_array(
                            episode_data, args.downsample_factor
                        )
                        all_data["states"][category][obj_name][state_type].append(
                            downsampled_data
                        )

        # Process wbc_step
        if "wbc_step" in store_in["data"]:
            episode_data = store_in["data/wbc_step"][ep_start:ep_end]
            downsampled_data = downsample_array(episode_data, args.downsample_factor)
            all_data["wbc_step"].append(downsampled_data)

        # Process wbc_target
        if "wbc_target" in store_in["data"]:
            for key in store_in["data/wbc_target"].keys():
                episode_data = store_in[f"data/wbc_target/{key}"][ep_start:ep_end]
                downsampled_data = downsample_array(
                    episode_data, args.downsample_factor
                )
                all_data["wbc_target"][key].append(downsampled_data)

        # Update episode end
        current_end += downsampled_length
        new_episode_ends.append(current_end)

    # Concatenate all episode data and save
    print("Concatenating and saving data...")

    # Save actions
    actions_data = np.concatenate(all_data["actions"], axis=0)
    chunk_size = get_optimal_chunks(shape=actions_data.shape, dtype=actions_data.dtype)
    data_group.create_dataset(
        "actions",
        data=actions_data,
        chunks=chunk_size,
        compressor=resolve_compressor("default"),
    )

    # Save observations
    obs_group = data_group.create_group("obs")
    for obs_key, obs_data in all_data["obs"].items():
        if isinstance(obs_data, dict):
            # Handle nested observations
            obs_sub_group = obs_group.create_group(obs_key)
            for sub_key, sub_data in obs_data.items():
                concat_data = np.concatenate(sub_data, axis=0)
                chunk_size = get_optimal_chunks(
                    shape=concat_data.shape, dtype=concat_data.dtype
                )
                obs_sub_group.create_dataset(
                    sub_key,
                    data=concat_data,
                    chunks=chunk_size,
                    compressor=resolve_compressor("default"),
                )
        else:
            # Handle flat observations
            concat_data = np.concatenate(obs_data, axis=0)
            chunk_size = get_optimal_chunks(
                shape=concat_data.shape, dtype=concat_data.dtype
            )
            obs_group.create_dataset(
                obs_key,
                data=concat_data,
                chunks=chunk_size,
                compressor=resolve_compressor("default"),
            )

    # Save states
    states_group = data_group.create_group("states")
    for category in ["articulation", "rigid_object"]:
        if category in all_data["states"] and all_data["states"][category]:
            cat_group = states_group.create_group(category)
            for obj_name, obj_data in all_data["states"][category].items():
                obj_group = cat_group.create_group(obj_name)
                for state_type, state_data in obj_data.items():
                    concat_data = np.concatenate(state_data, axis=0)
                    chunk_size = get_optimal_chunks(
                        shape=concat_data.shape, dtype=concat_data.dtype
                    )
                    obj_group.create_dataset(
                        state_type,
                        data=concat_data,
                        chunks=chunk_size,
                        compressor=resolve_compressor("default"),
                    )

    # Save wbc_step
    if all_data["wbc_step"]:
        wbc_step_data = np.concatenate(all_data["wbc_step"], axis=0)
        chunk_size = get_optimal_chunks(
            shape=wbc_step_data.shape, dtype=wbc_step_data.dtype
        )
        data_group.create_dataset(
            "wbc_step",
            data=wbc_step_data,
            chunks=chunk_size,
            compressor=resolve_compressor("default"),
        )

    # Save wbc_target
    if all_data["wbc_target"]:
        wbc_target_group = data_group.create_group("wbc_target")
        for key, data in all_data["wbc_target"].items():
            concat_data = np.concatenate(data, axis=0)
            chunk_size = get_optimal_chunks(
                shape=concat_data.shape, dtype=concat_data.dtype
            )
            wbc_target_group.create_dataset(
                key,
                data=concat_data,
                chunks=chunk_size,
                compressor=resolve_compressor("default"),
            )

    # Save new episode ends
    episode_ends_data = np.array(new_episode_ends)
    chunk_size = get_optimal_chunks(
        shape=episode_ends_data.shape, dtype=episode_ends_data.dtype
    )
    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends_data,
        chunks=chunk_size,
        compressor=resolve_compressor("default"),
    )

    print(f"Downsampling complete!")
    print(f"Original total timesteps: {episode_ends[-1] if num_episodes > 0 else 0}")
    print(
        f"Downsampled total timesteps: {new_episode_ends[-1] if new_episode_ends else 0}"
    )
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
