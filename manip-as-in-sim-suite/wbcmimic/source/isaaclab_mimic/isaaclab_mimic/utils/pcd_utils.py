# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Dict, Sequence, Tuple, Union

import isaaclab.utils.math as math_utils
import numpy as np
import torch
import warp as wp
from isaaclab.utils.array import TensorData, convert_to_torch


def create_pointcloud_from_depth_batch(
    intrinsic_matrix: np.ndarray | torch.Tensor | wp.array,
    depth: np.ndarray | torch.Tensor | wp.array,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(
        intrinsic_matrix, dtype=torch.float32, device=device
    )
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = math_utils.transform_points(depth_cloud, position, orientation)

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(
            torch.logical_and(~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)),
            dim=-1,
        )
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud


def create_pointcloud_from_rgbd_batch(
    intrinsic_matrix: torch.Tensor | np.ndarray,
    depth: torch.Tensor | np.ndarray,
    rgb: torch.Tensor | np.ndarray | Tuple[float, float, float] = None,
    normalize_rgb: bool = False,
    position: Sequence[float] | None | torch.Tensor = None,
    orientation: Sequence[float] | None | torch.Tensor = None,
    device: torch.device | str | None | torch.Tensor = None,
    num_channels: int = 3,
    maximum_distance: float = 100.0,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """Creates pointcloud from input depth image and camera transformation matrix.

    This function provides the same functionality as :meth:`create_pointcloud_from_depth` but also allows
    to provide the RGB values for each point.

    The ``rgb`` attribute is used to resolve the corresponding point's color:

    - If a ``np.array``/``wp.array``/``torch.tensor`` of shape (H, W, 3), then the corresponding channels encode RGB values.
    - If a tuple, then the point cloud has a single color specified by the values (r, g, b).
    - If None, then default color is white, i.e. (0, 0, 0).

    If the input ``normalize_rgb`` is set to :obj:`True`, then the RGB values are normalized to be in the range [0, 1].

    Args:
        intrinsic_matrix: A (3, 3) array/tensor providing camera's calibration matrix.
        depth: An array/tensor of shape (H, W) with values encoding the depth measurement.
        rgb: Color for generated point cloud. Defaults to None.
        normalize_rgb: Whether to normalize input rgb. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation `(w, x, y, z)` of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed. Defaults to None, in which case
            it takes the device that matches the depth image.
        num_channels: Number of channels in RGB pointcloud. Defaults to 3.

    Returns:
        A tuple of (N, 3) arrays or tensors containing the 3D coordinates of points and their RGB color respectively.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).
    """
    # check valid inputs
    if rgb is not None and not isinstance(rgb, tuple):
        if len(rgb.shape) == 3:
            rgb = rgb[None]
            if rgb.shape[2] not in [3, 4]:
                raise ValueError(
                    f"Input rgb image of invalid shape: {rgb.shape} != (H, W, 3) or (H, W, 4)."
                )
        elif len(rgb.shape) == 4:
            if rgb.shape[3] not in [3, 4]:
                raise ValueError(
                    f"Input rgb image of invalid shape: {rgb.shape}!= (H, W, 3) or (H, W, 4)."
                )
        else:
            raise ValueError(
                f"Input rgb image not three-dimensional. Received shape: {rgb.shape}."
            )
    if num_channels not in [3, 4]:
        raise ValueError(f"Invalid number of channels: {num_channels} != 3 or 4.")

    # check if input depth is numpy array
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    if is_numpy:
        depth = torch.from_numpy(depth).to(device=device)
    # retrieve XYZ pointcloud
    points_xyz = create_pointcloud_from_depth_batch(
        intrinsic_matrix, depth, True, position, orientation, device=device
    )

    # get image height and width
    im_height, im_width = depth.shape[-2:]
    # total number of points
    num_points = im_height * im_width
    # extract color value
    if rgb is not None:
        if isinstance(rgb, (np.ndarray, torch.Tensor)):
            # copy numpy array to preserve
            rgb = convert_to_torch(rgb, device=device, dtype=torch.float32)
            rgb = rgb[:, :, :, :3]
            # convert the matrix to (W, H, 3) from (H, W, 3) since depth processing
            # is done in the order (u, v) where u: (0, W-1) and v: (0 - H-1)
            batch_size, H, W, _ = rgb.shape

            # 使用 permute 重排维度，从 [B, H, W, 3] 变为 [B, W, H, 3]
            # 然后 reshape 为 [B, W*H, 3]
            points_rgb = rgb.permute(0, 2, 1, 3).reshape(batch_size, W * H, 3)
        elif isinstance(rgb, (tuple, list)):
            # same color for all points
            points_rgb = torch.Tensor(
                (rgb,) * num_points, device=device, dtype=torch.uint8
            )
        else:
            # default color is white
            points_rgb = torch.Tensor(
                ((0, 0, 0),) * num_points, device=device, dtype=torch.uint8
            )
    else:
        points_rgb = torch.Tensor(
            ((0, 0, 0),) * num_points, device=device, dtype=torch.uint8
        )
    # normalize color values
    if normalize_rgb:
        points_rgb = points_rgb.float() / 255

    # remove invalid points
    pts_idx_to_keep = torch.all(
        torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=-1
    )
    points_rgb[~pts_idx_to_keep, ...] = maximum_distance
    points_xyz[~pts_idx_to_keep, ...] = 0

    # add additional channels if required
    if num_channels == 4:
        points_rgb = torch.nn.functional.pad(
            points_rgb, (0, 1), mode="constant", value=1.0
        )

    # return everything according to input type
    if is_numpy:
        res = OrderedDict()
        res["pos"] = points_xyz.cpu().numpy()
        res["color"] = points_rgb.cpu().numpy()
        return res
    else:
        res = OrderedDict()
        res["pos"] = points_xyz
        res["color"] = points_rgb
        return res


def select_mask(obs, key, mask):
    if key in obs:
        obs[key] = obs[key][mask]


def pcd_filter_bound(cloud, eps=1e-3, max_dis=1.5, bound=None):
    # return (
    #     (pcd["pos"][..., 2] > eps)
    #     & (pcd["pos"][..., 1] < max_dis)
    #     & (pcd["pos"][..., 0] < max_dis)
    #     & (pcd["pos"][..., 2] < max_dis)
    # )
    if isinstance(cloud, dict):
        pc = cloud["pos"]  # (n, 3)
    else:
        assert isinstance(cloud, np.ndarray), f"{type(cloud)}"
        assert cloud.shape[1] == 3, f"{cloud.shape}"
        pc = cloud

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(
        np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z))
    )[0]

    return within_bound


def pcd_filter_with_mask(obs, mask):
    assert isinstance(obs, dict), f"{type(obs)}"
    for key in ["pos", "color", "seg", "visual_seg", "robot_seg"]:
        select_mask(obs, key, mask)


def pcd_downsample(
    obs,
    bound_clip=False,
    ground_eps=-1e-3,
    max_dis=15,
    num=1200,
    method="fps",
    bound=None,
):
    assert method in [
        "fps",
        "uniform",
    ], "expected method to be 'fps' or 'uniform', got {method}"

    sample_mehod = uniform_sampling if method == "uniform" else fps_sampling
    # import ipdb; ipdb.set_trace()
    if bound_clip:
        pcd_filter_with_mask(
            obs,
            pcd_filter_bound(obs, eps=ground_eps, max_dis=max_dis, bound=bound),
        )
    pcd_filter_with_mask(obs, sample_mehod(obs["pos"], num))
    return obs


def fps_sampling(points, npoints=1200):
    num_curr_pts = points.shape[0]
    if num_curr_pts < npoints:
        return np.random.choice(num_curr_pts, npoints, replace=True)
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    try:
        fps_idx = furthest_point_sample(points[..., :3], npoints)
    except:
        npoints = torch.tensor([npoints]).cuda()
        _, fps_idx = torch3d_ops.sample_farthest_points(points[..., :3], K=npoints)

    return fps_idx.squeeze(0).cpu().numpy()


def uniform_sampling(points, npoints=1200):
    n = points.shape[0]
    index = np.arange(n)
    if n == 0:
        return np.zeros(npoints, dtype=np.int64)
    if index.shape[0] > npoints:
        np.random.shuffle(index)
        index = index[:npoints]
    elif index.shape[0] < npoints:
        num_repeat = npoints // index.shape[0]
        index = np.concatenate([index for i in range(num_repeat)])
        index = np.concatenate([index, index[: npoints - index.shape[0]]])
    return index


def uniform_sampling_torch(points, npoints=1200):
    """
    均匀采样点云中的点（矩阵操作版本，无for循环）

    参数:
        points: torch.Tensor - 形状为[B, N, 3]或[N, 3]的点云
        npoints: int - 采样后的点数

    返回:
        torch.Tensor - 采样点的索引，形状为[B, npoints]或[npoints]
    """
    # 检查输入维度
    if len(points.shape) == 3:  # [B, N, 3]
        batch_size, n, _ = points.shape
        batch_mode = True
    elif len(points.shape) == 2:  # [N, 3]
        n = points.shape[0]
        batch_mode = False
        # 扩展为批处理模式以便统一处理
        points = points.unsqueeze(0)
        batch_size = 1
    else:
        raise ValueError(
            f"输入点云维度不正确，应为[B, N, 3]或[N, 3]，当前为{points.shape}"
        )

    # 处理空点云情况
    if n == 0:
        if batch_mode:
            return torch.zeros(
                (batch_size, npoints), dtype=torch.int64, device=points.device
            )
        else:
            return torch.zeros(npoints, dtype=torch.int64, device=points.device)

    # 创建索引张量 [B, N]
    indices = torch.arange(n, device=points.device).expand(batch_size, n)

    if n > npoints:
        # 使用矩阵操作进行随机采样
        # 为每个批次生成随机排列
        rand_indices = torch.argsort(
            torch.rand(batch_size, n, device=points.device), dim=1
        )
        # 选择前npoints个索引
        sampled_indices = torch.gather(indices, 1, rand_indices[:, :npoints])
    elif n < npoints:
        # 计算重复次数和剩余数量
        num_repeat = npoints // n
        remaining = npoints - num_repeat * n

        # 重复整个索引张量
        repeated_indices = indices.repeat_interleave(num_repeat, dim=1)

        # 添加剩余的索引
        if remaining > 0:
            remaining_indices = indices[:, :remaining]
            sampled_indices = torch.cat([repeated_indices, remaining_indices], dim=1)
        else:
            sampled_indices = repeated_indices
    else:
        # 如果点数正好等于npoints，直接返回索引
        sampled_indices = indices

    # 返回结果
    if batch_mode:
        return sampled_indices
    else:
        return sampled_indices[0]  # 移除批处理维度


def pcd_filter_bound_torch(pc, bound):
    """
    根据给定边界过滤点云（尽量避免循环的版本）

    参数:
        cloud: torch.Tensor或dict - 点云数据，形状为[B, N, 3]、[N, 3]或包含'pos'键的字典
        bound: list或torch.Tensor - 边界值 [x_min, x_max, y_min, y_max, z_min, z_max]
        eps: float - 最小高度阈值
        max_dis: float - 最大距离阈值

    返回:
        torch.Tensor - 在边界内的点的索引掩码，形状为[B, N]
        或
        list of torch.Tensor - 每个批次在边界内的点的索引
    """
    # 确保bound是tensor并且在正确的设备上
    if not isinstance(bound, torch.Tensor):
        bound = torch.tensor(bound, device=pc.device)

    # 检查输入维度
    batch_mode = len(pc.shape) == 3

    if not batch_mode:
        pc = pc.unsqueeze(0)  # [N, 3] -> [1, N, 3]

    # 确保bound是正确的形状
    if len(bound.shape) == 1:
        bound = bound.unsqueeze(0).expand(pc.shape[0], -1)

    # 计算边界条件
    within_bound_x = (pc[..., 0] > bound[:, 0:1]) & (pc[..., 0] < bound[:, 1:2])
    within_bound_y = (pc[..., 1] > bound[:, 2:3]) & (pc[..., 1] < bound[:, 3:4])
    within_bound_z = (pc[..., 2] > bound[:, 4:5]) & (pc[..., 2] < bound[:, 5:6])

    # 组合所有条件
    within_bound = within_bound_x & within_bound_y & within_bound_z

    # 两种返回方式：
    # 1. 返回掩码
    if batch_mode:
        return within_bound
    else:
        return within_bound[0]


def pcd_downsample_torch(
    obs,
    bound_clip=False,
    num=1200,
    method="uniform",
    bound=None,
):
    assert method in [
        "uniform",
    ], "expected method to be 'uniform', got {method}"

    if bound_clip:
        mask = pcd_filter_bound_torch(obs["pos"], bound=bound)
        obs = {k: [v[i][mask[i]] for i in range(len(v))] for k, v in obs.items()}
    res_obs = {k: [] for k in obs}
    for i in range(len(obs["pos"])):
        mask = uniform_sampling_torch(obs["pos"][i], npoints=num)
        for k in res_obs:
            res_obs[k].append(obs[k][i][mask])
    for k in res_obs:
        obs[k] = torch.stack(res_obs[k])
    return obs


if __name__ == "__main__":
    # Example usage
    intrinsic_matrix = np.eye(3)
    depth = np.random.rand(480, 640)
    pointcloud = create_pointcloud_from_depth_batch(intrinsic_matrix, depth)
    print(pointcloud.shape)  # Should print (N, 3) where N is the number of valid points
