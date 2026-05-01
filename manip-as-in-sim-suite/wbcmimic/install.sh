# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Set these environment variables to your installation paths
ISAACLAB_DIR=${ISAACLAB_DIR:-/path/to/IsaacLab}
CUROBO_DIR=${CUROBO_DIR:-/path/to/curobo}
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118 
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com 
$ISAACLAB_DIR/isaaclab.sh -i 
# install cuda
pip install -e $CUROBO_DIR --no-build-isolation
pip install -e source/bytemini_sim
pip install -e source/isaaclab_mimic