#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Resize image or depth arrays stored in zarr format with Hydra configuration support.
Supports multiple datasets with regex matching and parallel processing.
"""

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import hydra
import numpy as np
import zarr
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Set up logging
log = logging.getLogger(__name__)


def resize_single_dataset(
    zarr_path: str,
    input_key: str,
    output_key: str,
    target_height: int,
    target_width: int,
    chunk_frames: int = 10,
) -> Dict[str, Any]:
    """
    Resize images or depth maps in a single zarr array.

    Args:
        zarr_path: Path to the zarr file
        input_key: Input array key (e.g., '/data/obs/images/camera_0')
        output_key: Output array key (e.g., '/data/obs/images_resized/camera_0')
        target_height: Target height for resizing
        target_width: Target width for resizing
        chunk_frames: Number of frames to process at once

    Returns:
        Dictionary with processing results
    """
    try:
        # Open zarr file
        store = zarr.open(zarr_path, mode="r+")

        # Get input array
        if input_key not in store:
            raise ValueError(f"Input key '{input_key}' not found in zarr file")

        input_array = store[input_key]
        log.info(f"Processing {input_key} -> {output_key}")
        log.info(f"  Input shape: {input_array.shape}, dtype: {input_array.dtype}")

        # Get dimensions
        num_frames, orig_height, orig_width, channels = input_array.shape

        log.info(
            f"  Resizing from ({orig_height}, {orig_width}) to ({target_height}, {target_width})"
        )

        # Create output array with appropriate chunk size
        chunk_shape = (
            min(chunk_frames, num_frames),
            target_height,
            target_width,
            channels,
        )

        # Create output array
        output_array = store.create_dataset(
            output_key,
            shape=(num_frames, target_height, target_width, channels),
            chunks=chunk_shape,
            dtype=input_array.dtype,
            compressor=input_array.compressor,
            overwrite=True,
        )

        # Determine interpolation method
        if "depth" in input_key.lower():
            interpolation = cv2.INTER_NEAREST
        else:
            if target_height < orig_height or target_width < orig_width:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC

        # Process each chunk
        for chunk_idx in tqdm(range(0, num_frames, chunk_frames)):
            chunk_end = min(chunk_idx + chunk_frames, num_frames)
            chunk_data = input_array[chunk_idx:chunk_end]

            # Resize each frame in the chunk
            resized_chunk = np.zeros(
                (chunk_end - chunk_idx, target_height, target_width, channels),
                dtype=input_array.dtype,
            )

            for i in range(chunk_end - chunk_idx):
                frame = chunk_data[i]

                if channels == 1:
                    frame_2d = frame.squeeze()
                    resized_frame = cv2.resize(
                        frame_2d,
                        (target_width, target_height),
                        interpolation=interpolation,
                    )
                    resized_chunk[i] = resized_frame[..., np.newaxis]
                else:
                    resized_chunk[i] = cv2.resize(
                        frame,
                        (target_width, target_height),
                        interpolation=interpolation,
                    )

            # Write resized chunk
            output_array[chunk_idx:chunk_end] = resized_chunk

        # Calculate storage info
        storage_gb = (
            output_array.nbytes_stored / 1e9
            if hasattr(output_array, "nbytes_stored")
            else 0
        )

        return {
            "success": True,
            "input_key": input_key,
            "output_key": output_key,
            "num_frames": num_frames,
            "storage_gb": storage_gb,
        }

    except Exception as e:
        log.error(f"Error processing {input_key}: {str(e)}")
        return {
            "success": False,
            "input_key": input_key,
            "output_key": output_key,
            "error": str(e),
        }


def find_matching_keys(zarr_path: str, pattern: str) -> List[str]:
    """
    Find all keys in zarr file that match the given regex pattern.

    Args:
        zarr_path: Path to the zarr file
        pattern: Regex pattern to match keys

    Returns:
        List of matching keys
    """
    store = zarr.open(zarr_path, mode="r")
    all_keys = []

    def walk_zarr(group, prefix=""):
        for key, item in group.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(item, zarr.Array):
                all_keys.append(f"/{full_key}")
            elif isinstance(item, zarr.Group):
                walk_zarr(item, full_key)

    walk_zarr(store)

    # Filter keys based on pattern
    regex = re.compile(pattern)
    matching_keys = [key for key in all_keys if regex.match(key)]

    return matching_keys


def process_dataset_mapping(
    zarr_path: str,
    mapping: Dict[str, Any],
    target_height: int,
    target_width: int,
    chunk_frames: int,
) -> List[Dict[str, Any]]:
    """
    Process a single dataset mapping (can match multiple keys).

    Args:
        zarr_path: Path to the zarr file
        mapping: Dataset mapping configuration
        target_height: Target height
        target_width: Target width
        chunk_frames: Chunk size for processing

    Returns:
        List of processing results
    """
    results = []

    # Find all matching input keys
    input_pattern = mapping.get("input_pattern", mapping.get("input_key"))
    matching_keys = find_matching_keys(zarr_path, input_pattern)

    if not matching_keys:
        log.warning(f"No keys found matching pattern: {input_pattern}")
        return results

    log.info(f"Found {len(matching_keys)} keys matching pattern: {input_pattern}")

    for input_key in matching_keys:
        # Generate output key
        if "output_pattern" in mapping:
            # Apply regex substitution
            output_key = re.sub(
                mapping["input_pattern"], mapping["output_pattern"], input_key
            )
        else:
            # Use fixed output key
            output_key = mapping["output_key"]

        # Process single dataset
        result = resize_single_dataset(
            zarr_path, input_key, output_key, target_height, target_width, chunk_frames
        )
        results.append(result)

    return results


@hydra.main(
    version_base=None, config_path="../../config/tools", config_name="resize_zarr_example"
)
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration."""

    # Log configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Validate zarr paths
    zarr_paths = (
        cfg.zarr_paths
        if isinstance(cfg.zarr_paths, Iterable) and not isinstance(cfg.zarr_paths, str)
        else [cfg.zarr_paths]
    )

    for zarr_path in zarr_paths:
        if not Path(zarr_path).exists():
            log.error(f"Zarr file not found: {zarr_path}")
            continue

        log.info(f"\nProcessing zarr file: {zarr_path}")

        # Collect all tasks
        all_tasks = []
        for mapping in cfg.datasets:
            all_tasks.append((zarr_path, mapping))

        # Process datasets
        if cfg.parallel.enabled and len(all_tasks) > 1:
            # Parallel processing
            log.info(
                f"Processing {len(all_tasks)} dataset mappings in parallel (max_workers={cfg.parallel.max_workers})"
            )

            with ProcessPoolExecutor(max_workers=cfg.parallel.max_workers) as executor:
                future_to_task = {}

                for zarr_path, mapping in all_tasks:
                    future = executor.submit(
                        process_dataset_mapping,
                        zarr_path,
                        mapping,
                        cfg.resize.height,
                        cfg.resize.width,
                        cfg.processing.chunk_frames,
                    )
                    future_to_task[future] = (zarr_path, mapping)

                # Collect results
                all_results = []
                with tqdm(total=len(all_tasks), desc="Processing datasets") as pbar:
                    for future in as_completed(future_to_task):
                        results = future.result()
                        all_results.extend(results)
                        pbar.update(1)
        else:
            # Sequential processing
            log.info(f"Processing {len(all_tasks)} dataset mappings sequentially")
            all_results = []

            for zarr_path, mapping in tqdm(all_tasks, desc="Processing datasets"):
                results = process_dataset_mapping(
                    zarr_path,
                    mapping,
                    cfg.resize.height,
                    cfg.resize.width,
                    cfg.processing.chunk_frames,
                )
                all_results.extend(results)

        # Summary
        log.info("\n" + "=" * 50)
        log.info("Processing Summary:")
        log.info("=" * 50)

        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]

        log.info(f"Total processed: {len(all_results)}")
        log.info(f"Successful: {len(successful)}")
        log.info(f"Failed: {len(failed)}")

        if successful:
            total_storage = sum(r["storage_gb"] for r in successful)
            log.info(f"Total storage used: {total_storage:.2f} GB")

        if failed:
            log.info("\nFailed operations:")
            for r in failed:
                log.error(f"  {r['input_key']}: {r['error']}")


if __name__ == "__main__":
    main()
