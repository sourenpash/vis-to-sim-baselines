# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import collections
import os

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Circle, Rectangle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    required=True,
)
args = parser.parse_args()


zarr_store = zarr.DirectoryStore(os.path.expanduser(args.dataset_path))
zarr_root = zarr.group(store=zarr_store)
# 获取数据
rigid_pos = zarr_root["data"]["initial_state"]["rigid_object"]["bowl"]["root_pose"][
    :, :2
]
articulation_pos = zarr_root["data"]["initial_state"]["articulation"]["microwave"][
    "root_pose"
][:, :2]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制散点
ax.scatter(rigid_pos[:, 0], rigid_pos[:, 1], label="rigid body", color="r")
ax.scatter(
    articulation_pos[:, 0], articulation_pos[:, 1], color="b", label="articulation"
)

# 计算articulation位置的均值中心
art_center = np.mean(articulation_pos, axis=0)

# 绘制以articulation均值为中心的矩形框 (y方向±21.5cm，x方向±15cm)
rect_width = 0.30  # ±15cm
rect_height = 0.43  # ±21.5cm
rect = Rectangle(
    (art_center[0] - rect_width / 2, art_center[1] - rect_height / 2),
    rect_width,
    rect_height,
    linewidth=1,
    edgecolor="g",
    facecolor="none",
)
ax.add_patch(rect)

# 为每个rigid body位置绘制半径为3.5cm的圆
for pos in rigid_pos:
    circle = Circle((pos[0], pos[1]), 0.035, fill=False, edgecolor="r", linestyle="--")
    ax.add_patch(circle)

# 绘制以(0,0)为中心的矩形，x方向0.8，y方向1.2
center_rect = Rectangle(
    (-0.4, -0.6), 0.8, 1.2, linewidth=2, edgecolor="purple", facecolor="none"
)
ax.add_patch(center_rect)

# 绘制以(-6.25179097e-01, 4.98731775e-02)为中心，半径为0.1的圆
ur5_pos_circle = Circle(
    (-6.25179097e-01, 4.98731775e-02), 0.1, fill=False, edgecolor="orange", linewidth=2
)
ax.add_patch(ur5_pos_circle)

# Add legend and labels
ax.legend()
ax.set_xlabel("X axis (forward)")
ax.set_ylabel("Y axis (left)")
ax.set_title("Position Distribution Visualization")
ax.axis("equal")  # Ensure x and y axes have the same scale

# 保存图片
plt.savefig("pos_visualization.png")
plt.close(fig)
