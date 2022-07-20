#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch, sys
import torch.nn.functional as F
from panoptic_bev.utils.fake_ops import fake_linalg_inv, fake_repeat_interleave, fake_grid_sample
from panoptic_bev.custom.custom_repeat_interleave import custom_repeat_interleave
from panoptic_bev.custom.custom_inverse import custom_inverse
from panoptic_bev.custom.custom_grid_sample import custom_grid_sample

from panoptic_bev.utils import plogging
logger = plogging.get_logger()


@torch.jit.script
def _torch_inverse_cast(input: torch.Tensor) -> torch.Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    # if not isinstance(input, torch.Tensor):
    #     raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
    # dtype: torch.dtype = input.dtype
    # if dtype not in (torch.float32, torch.float64):
    dtype = torch.float32
    # print("custom_inverse, _torch_inverse_cast: {}".format(input.shape))
    return custom_inverse(input.to(dtype)).to(input.dtype)

@torch.jit.script
def normal_transform_pixel(
    in_size: torch.Tensor #(h,w)
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.
    """
    # tr_mat = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]])  # 3x3
    tr_mat_0 = torch.tensor([2.0, 0.0, 0.0])
    tr_mat_1 = torch.tensor([0.0, 2.0, 0.0])
    tr_mat_2 = torch.tensor([-1.0, -1.0, 1.0])

    # prevent divide by zero bugs
    # width_denom: float = eps if width == 1 else width - 1.0
    # height_denom: float = eps if height == 1 else height - 1.0
    width_denom = in_size[1] - 1 #width - 1.0
    height_denom = in_size[0] - 1 #height - 1.0

    tr_mat_0 = torch.div(tr_mat_0, width_denom)
    tr_mat_1 = torch.div(tr_mat_1, height_denom)
    tr_mat = torch.stack([tr_mat_0, tr_mat_1, tr_mat_2], dim=1)

    return tr_mat.unsqueeze(0)  # 1x3x3

@torch.jit.script
def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: torch.Tensor, dsize_dst: torch.Tensor
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).

    Returns:
        the normalized homography of shape :math:`(B, 3, 3)`.
    """
    # source and destination sizes
    # src_h, src_w = dsize_src[0], dsize_src[1]
    # dst_h, dst_w = dsize_dst[0], dsize_dst[1]

    # compute the transformation pixel/norm for src/dst
    # src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(dsize_src).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)

    # dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dsize_dst).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm

@torch.jit.script
def create_meshgrid(
    src: torch.Tensor,
    in_size: torch.Tensor,
    normalized_coordinates: bool = True,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    height = in_size[0]
    width = in_size[1]

    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=src.device, dtype=src.dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=src.device, dtype=src.dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

@torch.jit.script
def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)

@torch.jit.script
def convert_points_from_homogeneous(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = torch.tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    # we check for points at max_val
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]

@torch.jit.script
def transform_points(trans_01: torch.Tensor, points_1: torch.Tensor) -> torch.Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """
    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    # shape_inp = list(points_1.shape), to avoid Sequence ops.
    shape_inp = points_1.shape
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    # rpts = torch.div(points_1.shape[0], trans_01.shape[0], rounding_mode='floor').to(trans_01.device)
    rpts = int(points_1.shape[0])# // trans_01.shape[0]
    trans_01 = custom_repeat_interleave(trans_01, repeats=rpts, dim=0)

    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # print("transform_points: shape_inp: {}, points_0: {}".format(shape_inp, points_0.shape))
    # reshape to the input shape, keep shape_inp unchanged, to avoid squence ops, anyway shape_inp is unchanged at all!
    assert shape_inp[-2] == points_0.shape[-2]
    assert shape_inp[-1] == points_0.shape[-1]
    # shape_inp[-2] = points_0.shape[-2]
    # shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0

@torch.jit.script
def warp_perspective(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: torch.Tensor,
) -> torch.Tensor:

    B, _, H, W = src.size()
    # h_out, w_out = dsize[0], dsize[1]

    # we normalize the 3x3 transformation matrix and convert to 3x4
    # dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, torch.tensor(src.shape[2:]), dsize)  # Bx3x3

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm).to(src.device)  # Bx3x3

    # logger.info("[warp_perspective] M: {}, src: {}, dst: {}, src_norm_trans_dst_norm: {}".format(M, src.shape[2:], dsize, src_norm_trans_dst_norm))

    # this piece of code substitutes F.affine_grid since it does not support 3x3
    # grid = (create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device).to(src.dtype).repeat(B, 1, 1, 1))
    grid = (create_meshgrid(src, dsize, normalized_coordinates=True).repeat(B, 1, 1, 1))

    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    # if padding_mode == "fill":
    #     return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)

    # return fake_grid_sample(src, grid)
    return F.grid_sample(src, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
    # return custom_grid_sample(src, grid)

@torch.jit.script
def warp_perspective_v2(
    src: torch.Tensor,
    src_norm_trans_dst_norm: torch.Tensor,
    dsize: torch.Tensor,
) -> torch.Tensor:

    B, _, H, W = src.size()
    # h_out, w_out = dsize[0], dsize[1]

    # this piece of code substitutes F.affine_grid since it does not support 3x3
    # grid = (create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device).to(src.dtype).repeat(B, 1, 1, 1))
    grid = (create_meshgrid(src, dsize, normalized_coordinates=True).repeat(B, 1, 1, 1))

    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    # if padding_mode == "fill":
    #     return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)

    # return fake_grid_sample(src, grid)
    return F.grid_sample(src, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
