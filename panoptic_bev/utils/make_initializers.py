#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch, sys
import torch.nn.functional as F
from typing import Tuple, List
from panoptic_bev.utils.transformer_ts import get_init_homography
from panoptic_bev.custom.custom_inverse import custom_inverse
from panoptic_bev.utils.kornia_geometry_onnx import normalize_homography, _torch_inverse_cast

from panoptic_bev.utils import plogging
logger = plogging.get_logger()

class InitializerGenerator(object):
    g_intrinsics_list = None
    g_input_image_shape = None
    g_output_bev_shape = None
    g_theta_ipm = []
    g_theta_ipm_inv = []
    g_src_norm_trans_dst_norm = None
    g_src_norm_trans_dst_norm_inv = None

    def __init__(self, intrisics_list:torch.Tensor, input_image_shape:torch.Tensor, output_bev_shape:torch.Tensor):
        self.g_intrinsics_list = intrisics_list
        self.g_input_image_shape = input_image_shape
        self.g_output_bev_shape = output_bev_shape

    def __init__(self, dataloader):
        self.load_intrinsics(dataloader)

    def __del__(self):
        pass

    # using first batch intrinsics
    def load_intrinsics(self, dataloader):
        sample = dataloader.__getitem__(0)
        # sample["calib"]: (6,3,3), g_intrinsics_list: (1,6,3,3)
        self.g_intrinsics_list = sample["calib"].unsqueeze(0)
        logger.info("self.g_intrinsics_list: {}".format(self.g_intrinsics_list))

    def get_intrinsic(self, index):
        # should be (1,3,3)
        return self.g_intrinsics_list[:, index]

    def get_intrinsics_list(self):
        return self.g_intrinsics_list

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
        dsize_src = torch.tensor([int(input_image_shape[0]*img_scale), int(input_image_shape[1]*img_scale)])
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
    input_image_shape = torch.tensor([448, 768], dtype=torch.int)
    output_bev_shape = torch.tensor([768, 896], dtype=torch.int)
    # gen = InitializerGenerator(intrisics_list, input_image_shape, output_bev_shape)
    gen = InitializerGenerator()

if __name__ == "__main__":
    test()