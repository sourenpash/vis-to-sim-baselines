#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Simple script to downsample episodes in a zarr dataset.
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm


def process_array_or_group(source, target_parent, key, factor, episode_indices):
    """Process a zarr array or group, downsampling if needed."""
    source_path = f"{source.path}/{key}" if hasattr(source, "path") else key

    if isinstance(source[key], zarr.Group):
        # Create corresponding group in target
        target_group = target_parent.create_group(key)
        # Recursively process all items in the group
        for sub_key in source[key].keys():
            process_array_or_group(
                source[key], target_group, sub_key, factor, episode_indices
            )
    else:
        # It's an array
        data = source[key][:]

        # Check if this is episode data that needs downsampling
        if (
            len(data.shape) > 0
            and len(episode_indices) > 0
            and data.shape[0] == episode_indices[-1]
        ):
            # This is timestep data - needs downsampling
            print(f"  Downsampling {source_path}: {data.shape}")

            # Collect downsampled data for all episodes
            all_downsampled = []
            for i, (start, end) in enumerate(episode_indices):
                episode_data = data[start:end]
                # Downsample by taking every 'factor' timesteps
                downsampled = episode_data[::factor]
                all_downsampled.append(downsampled)

            # Concatenate all episodes
            result = np.concatenate(all_downsampled, axis=0)
            target_parent[key] = result
            print(f"    -> {result.shape}")
        else:
            # This is metadata or per-episode data - copy as is
            print(f"  Copying {source_path}: {data.shape}")
            target_parent[key] = data


def main():
    parser = argparse.ArgumentParser(
        description="Downsample episodes in a zarr dataset"
    )
    parser.add_argument("--input", type=str, required=True, help="Input zarr file")
    parser.add_argument("--output", type=str, required=True, help="Output zarr file")
    parser.add_argument(
        "--factor", type=int, default=2, help="Downsample factor (default: 2)"
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None, help="Process only first N episodes"
    )
    args = parser.parse_args()

    print(f"Opening {args.input}...")
    store_in = zarr.open(args.input, mode="r")

    # Create output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store_out = zarr.open(args.output, mode="w")

    # Get episode information
    episode_ends = store_in["meta/episode_ends"][:]
    num_episodes = len(episode_ends)

    if args.max_episodes:
        num_episodes = min(num_episodes, args.max_episodes)
        episode_ends = episode_ends[:num_episodes]

    print(f"\nProcessing {num_episodes} episodes with downsample factor {args.factor}")

    # Calculate episode boundaries
    episode_indices = []
    for i in range(num_episodes):
        start = 0 if i == 0 else episode_ends[i - 1]
        end = episode_ends[i]
        episode_indices.append((start, end))

    # Process data group
    print("\nProcessing data...")
    data_out = store_out.create_group("data")
    for key in store_in["data"].keys():
        process_array_or_group(
            store_in["data"], data_out, key, args.factor, episode_indices
        )

    # Process meta group (mostly copy as-is, but update episode_ends)
    print("\nProcessing metadata...")
    meta_out = store_out.create_group("meta")
    for key in store_in["meta"].keys():
        if key == "episode_ends":
            # Recalculate episode ends after downsampling
            new_ends = []
            total = 0
            for start, end in episode_indices:
                ep_length = end - start
                downsampled_length = len(range(0, ep_length, args.factor))
                total += downsampled_length
                new_ends.append(total)
            meta_out[key] = np.array(new_ends)
            print(f"  Updated episode_ends: {len(new_ends)} episodes")
        else:
            # Copy other metadata as-is
            if isinstance(store_in["meta"][key], zarr.Group):
                process_array_or_group(
                    store_in["meta"], meta_out, key, args.factor, episode_indices
                )
            else:
                data = store_in["meta"][key][:]
                # Truncate to num_episodes if needed
                if len(data.shape) > 0 and data.shape[0] > num_episodes:
                    data = data[:num_episodes]
                meta_out[key] = data
                print(f"  Copied meta/{key}: {data.shape}")

    print(f"\nDone! Output saved to {args.output}")
    print(f"Original timesteps: {episode_ends[-1] if episode_ends.size > 0 else 0}")
    print(f"Downsampled timesteps: {new_ends[-1] if new_ends else 0}")


if __name__ == "__main__":
    main()
