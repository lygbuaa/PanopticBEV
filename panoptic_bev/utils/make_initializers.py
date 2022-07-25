#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch, sys
import torch.nn.functional as F
from typing import Tuple, List
from panoptic_bev.utils.transformer_ts import get_init_homography
from panoptic_bev.custom.custom_inverse import custom_inverse
from panoptic_bev.custom.custom_affine_grid import custom_affine_grid
from panoptic_bev.utils.kornia_geometry_onnx import normalize_homography, _torch_inverse_cast

from panoptic_bev.utils import plogging
logger = plogging.get_logger()

class InitializerGenerator(object):
    g_intrinsics_list = None
    g_extrinsics_list = None
    g_valid_msk_list = None
    g_input_image_shape = None
    g_output_bev_shape = None

    def __init__(self, intrisics_list:torch.Tensor, input_image_shape:torch.Tensor, output_bev_shape:torch.Tensor):
        self.g_intrinsics_list = intrisics_list
        self.g_input_image_shape = input_image_shape
        self.g_output_bev_shape = output_bev_shape

    def __init__(self, dataloader):
        self.load_params_from_dataset(dataloader)

    def __del__(self):
        pass

    # using first batch intrinsics
    def load_params_from_dataset(self, dataloader):
        sample = dataloader.__getitem__(0)
        # sample["calib"]: (6,3,3), g_intrinsics_list: (1,6,3,3)
        self.g_intrinsics_list = sample["calib"].unsqueeze(0)
        logger.info("self.g_intrinsics_list: {}".format(self.g_intrinsics_list))
        # sample["extrinsics"]: (6,2,3), g_extrinsics_list: (1,6,2,3)
        self.g_extrinsics_list = sample["extrinsics"].unsqueeze(0)
        logger.info("self.g_extrinsics_list: {}".format(self.g_extrinsics_list))
        # sample["valid_msk"]: (6,896,768), g_valid_msk_list: (1,6,896,768)
        self.g_valid_msk_list = sample["valid_msk"].unsqueeze(0)
        logger.info("self.g_valid_msk_list: {}".format(self.g_valid_msk_list.shape))

    def get_intrinsic(self, index):
        # should be (1,3,3)
        return self.g_intrinsics_list[:, index]

    def get_intrinsics_list(self):
        return self.g_intrinsics_list

    def get_extrinsics_list(self):
        return self.g_extrinsics_list

    def get_valid_msk_list(self):
        return self.g_valid_msk_list

    def generate_theta_ipm_list(self, extrinsics, px_per_metre, img_scale, out_img_size_reverse):
        b, n, _, _ = self.g_intrinsics_list.shape
        theta_ipm_list = []
        theta_ipm_inv_list = []
        for idx in range(n):
            intrinsics = self.g_intrinsics_list[0, idx]
            theta_ipm_i = get_init_homography(intrinsics, extrinsics, px_per_metre, img_scale, out_img_size_reverse).view(-1, 3, 3)
            theta_ipm_list.append(theta_ipm_i)
            theta_ipm_inv_i = custom_inverse(theta_ipm_i)
            theta_ipm_inv_list.append(theta_ipm_inv_i)
            logger.info("[{}] theta_ipm_i: {}, theta_ipm_inv_i: {}".format(idx, theta_ipm_i, theta_ipm_inv_i))
        # self.g_theta_ipm_list = theta_ipm_list
        # self.g_theta_ipm_inv_list = theta_ipm_inv_list
        return theta_ipm_list, theta_ipm_inv_list

    def generate_src_norm_trans_dst_norm(self, theta_ipm_list, theta_ipm_inv_list, img_scale, input_image_shape:Tuple, output_bev_shape:Tuple):
        n = len(theta_ipm_list)
        # dsize_src needs scale down
        dsize_src = torch.tensor([int(input_image_shape[0]*img_scale), int(input_image_shape[1]*img_scale)])
        # dsize_dst already scaled down by FlatTransformer
        dsize_dst = torch.tensor([int(output_bev_shape[0]), int(output_bev_shape[1])])
        logger.info("generate_src_norm_trans_dst_norm, img_scale: {}, dsize_src: {}, dsize_dst: {}".format(img_scale, dsize_src, dsize_dst))
        src_norm_trans_dst_norm_list = []
        src_norm_trans_dst_norm_inv_list = []
        for idx in range(n):
            theta_ipm = theta_ipm_list[idx]
            theta_ipm_inv = theta_ipm_inv_list[idx]
            # logger.info("[{}] theta_ipm: {}, theta_ipm_inv: {}".format(idx, theta_ipm, theta_ipm_inv))
            dst_norm_trans_src_norm = normalize_homography(theta_ipm, dsize_src, dsize_dst)
            src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
            src_norm_trans_dst_norm_list.append(src_norm_trans_dst_norm)

            dst_norm_trans_src_norm_inv = normalize_homography(theta_ipm_inv, dsize_dst, dsize_src)
            src_norm_trans_dst_norm_inv = _torch_inverse_cast(dst_norm_trans_src_norm_inv)
            src_norm_trans_dst_norm_inv_list.append(src_norm_trans_dst_norm_inv)

            logger.info("[{}] src_norm_trans_dst_norm: {}, src_norm_trans_dst_norm_inv: {}".format(idx, src_norm_trans_dst_norm, src_norm_trans_dst_norm_inv))
        return src_norm_trans_dst_norm_list, src_norm_trans_dst_norm_inv_list

    # grid_size = feat.size()
    def make_affine_grid(self, angle, tx, ty, grid_h, grid_w):
        angle = torch.deg2rad(angle)
        theta = torch.stack([torch.stack([torch.cos(angle), torch.sin(-angle), tx]), torch.stack([torch.sin(angle), torch.cos(angle), ty])], dim=0)
        grid = custom_affine_grid(theta=theta.unsqueeze(0), N=1, H=grid_h, W=grid_w, align_corners=False)
        return grid

    # output_bev_shape = (self.Z_out, self.W_out)
    def generate_valid_mask(self, z_out, w_out):
        dsize_msk = ([int(w_out), int(z_out)])
        valid_mask_list = []
        b, n, h, w = self.g_valid_msk_list.shape
        for idx in range(n):
            msk_t = self.g_valid_msk_list[0:1, idx:idx+1, :, :]
            msk_t = F.interpolate(msk_t, dsize_msk, mode="nearest")
            valid_mask_list.append(msk_t)
            logger.info("[{}] msk_t: {}".format(idx, msk_t.shape))
        return valid_mask_list

    def generate_affine_grid(self, z_out, w_out, bev_resolution):
        b, n, _, _ = self.g_extrinsics_list.shape
        # dsize_grid = torch.Size([int(w_out), int(z_out)])
        grid_h=int(w_out)
        # double grid_w due to "feat_merged = F.pad(feat_merged, (W, 0), mode="constant", value=0)"
        grid_w=int(z_out*2)
        affine_grid_list = []
        for idx in range(n):
            extrinsics = self.g_extrinsics_list[0, idx]
            ccw_angle = extrinsics[1][0]
            # if keep bev output as [896, 768]
            # tx = -1.0 - extrinsics[0][0]/self.BEV_RESOLUTION/self.Z_out
            # ty = 0.0 + extrinsics[0][1]/self.BEV_RESOLUTION/self.W_out
            # if double bev output to [896, 768*2]
            tx = -1 * extrinsics[0][0]/bev_resolution/z_out/2
            ty = extrinsics[0][1]/bev_resolution/w_out
            grid = self.make_affine_grid(angle=ccw_angle, tx=tx, ty=ty, grid_h=grid_h, grid_w=grid_w)
            affine_grid_list.append(grid)
            logger.info("[{}] affine grid: {}".format(idx, grid.shape))
        return affine_grid_list

def test():
    intrisics_list = torch.tensor(
        [[[607.8802,   0.0000, 391.8082],
         [  0.0000, 630.3943, 244.6613],
         [  0.0000,   0.0000,   1.0000]],

        [[610.8470,   0.0000, 396.7754],
         [  0.0000, 633.4709, 238.8097],
         [  0.0000,   0.0000,   1.0000]],

        [[603.2359,   0.0000, 380.2140],
         [  0.0000, 625.5779, 245.2928],
         [  0.0000,   0.0000,   1.0000]],

        [[388.4261,   0.0000, 398.0254],
         [  0.0000, 402.8122, 239.8186],
         [  0.0000,   0.0000,   1.0000]],

        [[604.5667,   0.0000, 387.4814],
         [  0.0000, 626.9580, 249.4841],
         [  0.0000,   0.0000,   1.0000]],

        [[605.2068,   0.0000, 387.8248],
         [  0.0000, 627.6218, 246.5665],
         [  0.0000,   0.0000,   1.0000]]])

    extrinsics_list = torch.tensor(
        [[[[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00]],

         [[-1.7691e-01,  4.7869e-01, -1.6294e-03],
          [ 5.4835e+01,  3.5954e+02,  1.6756e-01]],

         [[-6.6510e-01,  4.6885e-01,  8.0013e-02],
          [ 1.0827e+02,  5.9413e-01,  3.5983e+02]],

         [[-1.6725e+00, -1.2494e-02,  6.8146e-02],
          [ 1.7953e+02,  3.5872e+02,  2.7536e-01]],

         [[-6.8591e-01, -4.9651e-01,  5.1438e-02],
          [ 2.4889e+02,  6.0879e-01,  6.6530e-01]],

         [[-1.4994e-01, -5.0935e-01, -1.5210e-02],
          [ 3.0328e+02,  4.5877e-01,  5.6502e-01]]]]
    )
    input_image_shape = torch.tensor([448, 768], dtype=torch.int)
    output_bev_shape = torch.tensor([768, 896], dtype=torch.int)
    # gen = InitializerGenerator(intrisics_list, input_image_shape, output_bev_shape)
    gen = InitializerGenerator()

if __name__ == "__main__":
    test()