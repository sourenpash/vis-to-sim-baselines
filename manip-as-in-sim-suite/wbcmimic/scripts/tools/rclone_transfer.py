#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Rclone transfer script
"""

import argparse
import os
import re
import subprocess
import sys
import time

import wandb


def main():
    parser = argparse.ArgumentParser(
        description="Transfer files using rclone with progress tracking and notifications"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Source path for rclone transfer"
    )
    parser.add_argument(
        "--dest", type=str, required=True, help="Destination path for rclone transfer"
    )
    parser.add_argument(
        "--transfers",
        type=int,
        default=48,
        help="Number of parallel transfers (default: 48)",
    )

    args = parser.parse_args()

    wandb.init(
        project="manip_as_in_sim",
        entity="your_entity",
        name="rclone_transfer_run",
    )

    # Build rclone command
    rclone_cmd = [
        "rclone",
        "copy",
        "--transfers",
        str(args.transfers),
        "--progress",
        args.source,
        args.dest,
    ]

    print(f"Executing rclone command: {' '.join(rclone_cmd)}")

    try:
        start_time = time.time()

        # Execute rclone command
        process = subprocess.Popen(
            rclone_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Pattern to match rclone progress output
        progress_pattern = re.compile(r"Transferred:\s+(.+?),\s+(\d+)%")

        for line in process.stdout:
            print(line, end="")


            match = progress_pattern.search(line)
            if match:
                transferred = match.group(1)
                percentage = int(match.group(2))
                wandb.log(
                    {
                        "transfer_progress": percentage,
                        "transferred_data": transferred,
                    }
                )

        process.wait()

        if process.returncode == 0:
            end_time = time.time()
            duration = end_time - start_time

            print(f"Transfer completed successfully in {duration:.2f} seconds")
        else:
            print(f"Transfer failed with return code: {process.returncode}")
            sys.exit(process.returncode)

    except subprocess.CalledProcessError as e:
        print(f"Error executing rclone: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("Transfer interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
