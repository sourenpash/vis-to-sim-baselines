# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

if __name__ == "__main__":
    print(PROJECT_ROOT)
