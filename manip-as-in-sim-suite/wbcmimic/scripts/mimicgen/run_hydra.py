#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Hydra version of wrapper script to launch generate_dataset_parallel_all.py.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from threading import Thread

import hydra
import wandb
from omegaconf import DictConfig


@hydra.main(
    config_path="../..", config_name="config/mimicgen/generate_data", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Main function to launch mimicgen data generation with Hydra configuration."""

    print(f"Running with configuration: {cfg}")

    # Initialize wandb
    if cfg.logging.enabled:
        wandb.init(project=cfg.logging.project, name=cfg.logging.name)

    # Build command to execute the target script
    target_script = os.path.join(
        os.path.dirname(__file__), "generate_dataset_parallel_all.py"
    )
    cmd = [sys.executable, target_script]

    # Add all configuration arguments
    if cfg.task is not None:
        cmd.extend(["--task", cfg.task])
    if cfg.generation_num_trials is not None:
        cmd.extend(["--generation_num_trials", str(cfg.generation_num_trials)])
    cmd.extend(["--num_envs", str(cfg.num_envs)])
    cmd.extend(["--input_file", cfg.input_file])
    cmd.extend(["--output_file", cfg.output_file])
    if cfg.pause_subtask:
        cmd.append("--pause_subtask")
    if cfg.distributed:
        cmd.append("--distributed")
    cmd.extend(["--n_procs", str(cfg.n_procs)])
    if cfg.state_only:
        cmd.append("--state_only")
    if cfg.record_all:
        cmd.append("--record_all")
    cmd.extend(["--seed", str(cfg.seed)])
    cmd.extend(["--mimic_mode", cfg.mimic_mode])
    cmd.extend(["--failed_save_prob", str(cfg.failed_save_prob)])
    if cfg.batch_splits is not None:
        cmd.extend(["--batch_splits"] + [str(x) for x in cfg.batch_splits])
    if cfg.camera_names:
        cmd.extend(["--camera_names"] + [str(name) for name in cfg.camera_names])
    for key, value in cfg.get("extra_args", {}).items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    if cfg.wbc_solver_cfg is not None:
        wbc_cfgs = dict(cfg.wbc_solver_cfg)
        temp_json_path = Path(tempfile.gettempdir()) / "extra_wbc_params.json"
        with open(temp_json_path, "w") as f:
            json.dump(wbc_cfgs, f)
        cmd.extend(["--extra_wbc_params_path", str(temp_json_path)])
    if cfg.no_save_images:
        cmd.append("--no_save_images")

    # Print the command being executed for debugging
    print(f"Executing command: {' '.join(cmd)}")

    # Setup monitoring
    last_time = None

    def timer():
        nonlocal last_time
        while True:
            if (
                last_time is not None
                and time.time() - last_time > cfg.monitoring.timeout
            ):
                # you can do something here if the timeout is reached
                # This ofter indicates that the generation process is stuck
                last_time = time.time()
            time.sleep(cfg.monitoring.check_interval)

    # Setup batch file monitoring and transfer
    transferred_batches = set()
    transfer_lock = threading.Lock()

    def monitor_and_transfer_batches():
        """Monitor for completed batch files and transfer them immediately."""
        if cfg.batch_splits is None:
            return

        base_name, ext = os.path.splitext(cfg.output_file)

        while True:
            try:
                # Check for .done files
                for i in range(len(cfg.batch_splits)):
                    with transfer_lock:
                        if i in transferred_batches:
                            continue

                    batch_filename = f"{base_name}.{i+1:03d}{ext}"
                    done_file_path = batch_filename + ".done"

                    if os.path.exists(done_file_path):
                        # Found a completed batch
                        print(f"\nDetected completed batch {i+1}, starting transfer...")

                        # Transfer the batch file - use absolute paths
                        batch_abs_path = os.path.abspath(batch_filename)
                        batch_target = os.path.join(
                            cfg.file_transfer.remote_base_dir,
                            os.path.relpath(
                                batch_abs_path, cfg.file_transfer.local_base_dir
                            ),
                        )
                        rclone_cmd = [
                            "rclone",
                            "copy",
                            "--transfers",
                            str(cfg.file_transfer.transfers),
                            batch_abs_path,
                            os.path.dirname(batch_target),
                        ]

                        try:
                            subprocess.run(rclone_cmd, check=True)
                            print(
                                f"Batch {i+1} transferred successfully to {batch_target}"
                            )

                            with transfer_lock:
                                transferred_batches.add(i)

                            # Delete local files if requested
                            if cfg.file_transfer.delete_after_transfer:
                                os.remove(batch_abs_path)
                                os.remove(done_file_path)
                                print(f"Deleted local files for batch {i+1}")

                        except subprocess.CalledProcessError as e:
                            print(f"Error transferring batch {i+1}: {e}")

            except Exception as e:
                print(f"Error in batch monitor: {e}")

            time.sleep(5)  # Check every 5 seconds

    # Start batch monitoring thread if using batch splits
    if cfg.batch_splits is not None:
        batch_monitor_thread = Thread(target=monitor_and_transfer_batches, daemon=True)
        batch_monitor_thread.start()

    # Execute the target script with all arguments
    monitor_thread = Thread(target=timer, daemon=True)
    monitor_thread.start()
    episode_pattern = re.compile(r"Write episode (\d+) took ([\d.]+) seconds")
    target_num = int(cfg.n_procs) * int(cfg.generation_num_trials)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        try:
            for line in process.stdout:
                match = episode_pattern.search(line)
                if match:
                    last_time = time.time()

                print(line, end="")

        except KeyboardInterrupt:
            print("Stopping data collection...")

        # Wait for process to complete
        process.wait()
        print("Data generation process completed.")

        # Final cleanup and transfer of any remaining files
        output_file_path = cfg.output_file
        target_file_path = os.path.join(
            cfg.file_transfer.remote_base_dir,
            os.path.relpath(output_file_path, cfg.file_transfer.local_base_dir),
        )

        # If batch_splits is used, check for any untransferred files
        if cfg.batch_splits is not None:
            base_name, ext = os.path.splitext(output_file_path)
            untransferred_files = []

            # Find any remaining batch files
            with transfer_lock:
                for i in range(len(cfg.batch_splits)):
                    if i not in transferred_batches:
                        batch_filename = f"{base_name}.{i+1:03d}{ext}"
                        if os.path.exists(batch_filename):
                            untransferred_files.append((i, batch_filename))

            if untransferred_files:
                print(
                    f"\nFound {len(untransferred_files)} untransferred batch files, transferring now..."
                )
                for i, batch_file in untransferred_files:
                    batch_target = os.path.join(
                        cfg.file_transfer.remote_base_dir,
                        os.path.relpath(batch_file, cfg.file_transfer.local_base_dir),
                    )
                    rclone_cmd = [
                        "rclone",
                        "copy",
                        "--transfers",
                        str(cfg.file_transfer.transfers),
                        "--progress",
                        batch_file,
                        os.path.dirname(batch_target),
                    ]
                    print(f"Transferring remaining batch {i+1} to {batch_target}")
                    try:
                        subprocess.run(rclone_cmd, check=True)
                        print(f"Batch {i+1} transferred successfully")

                        # Delete local files immediately after successful transfer if requested
                        if cfg.file_transfer.delete_after_transfer:
                            os.remove(batch_file)
                            print(f"Deleted local file: {batch_file}")
                            done_file = batch_file + ".done"
                            if os.path.exists(done_file):
                                os.remove(done_file)
                                print(f"Deleted done file: {done_file}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error transferring batch {i+1}: {e}")

            print(
                f"\nAll batches processed. Total transferred: {len(transferred_batches) + len(untransferred_files)}/{len(cfg.batch_splits)}"
            )
        else:
            # Single file transfer
            if os.path.exists(output_file_path):
                rclone_cmd = [
                    "rclone",
                    "copy",
                    "--transfers",
                    str(cfg.file_transfer.transfers),
                    "--progress",
                    output_file_path,
                    os.path.dirname(target_file_path),
                ]
                print(f"Executing rclone command: {' '.join(rclone_cmd)}")
                subprocess.run(rclone_cmd, check=True)

                if cfg.file_transfer.delete_after_transfer:
                    os.remove(output_file_path)

        sys.exit(0)

    except subprocess.CalledProcessError as e:
        print(f"Error executing generate_dataset_parallel_all.py: {e}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Ensure subprocess termination
        if process and process.poll() is None:
            print("Terminating subprocess...")
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    print("Process did not terminate, sending SIGKILL...")
                    process.kill()
            except Exception as e:
                print(f"Error while terminating subprocess: {e}")


if __name__ == "__main__":
    main()
