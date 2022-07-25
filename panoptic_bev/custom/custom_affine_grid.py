#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch
from typing import Tuple, List

@torch.jit.script
def linspace_from_neg_one(theta:torch.Tensor, num_steps:int):
    rr = torch.linspace(-1, 1, num_steps, dtype=theta.dtype, device=theta.device)
    rr = rr * (num_steps-1)/num_steps
    return rr[0:num_steps]

@torch.jit.script
def custom_affine_grid(theta:torch.Tensor, N:int, H:int, W:int, align_corners:bool=False):
    assert N == 1, 'only support N=1!'
    # return torch.rand(N, H, W, 2, dtype=torch.float, device=theta.device)
    base_grid = torch.empty(size=[N, H, W, 3], dtype=theta.dtype, device=theta.device)
    base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W))
    base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H).unsqueeze_(-1))
    base_grid.select(-1, 2).fill_(1)

    grid = base_grid.view(N, H*W, 3).bmm(theta.transpose(1, 2))
    return grid.view(N, H, W, 2)

def test():
    angle = torch.tensor(10.0/57.3)
    tx = torch.tensor(0.5)
    ty = torch.tensor(0.0)
    theta = torch.stack([torch.stack([torch.cos(angle), torch.sin(-angle), tx]),torch.stack([torch.sin(angle), torch.cos(angle), ty])], dim=0).unsqueeze(0)
    # feat = torch.rand(size=[1, 256, 24, 28])
    feat = torch.rand(size=[1, 3, 5, 5])
    print("theta: {}, feat: {}".format(theta.shape, feat.shape))

    grid1 = torch.nn.functional.affine_grid(theta, feat.size(), align_corners=False)
    print("torch.affine_grid: {}\n {}".format(grid1.shape, grid1))

    N, C, H, W = feat.shape
    grid2 = custom_affine_grid(theta, N=N, H=H, W=W, align_corners=False)
    print("custom_affine_grid: {}\n {}".format(grid2.shape, grid2))

    diff = grid2 - grid1
    print("diff: \n{}".format(diff))

    # torch.onnx.export(
    #     model=custom_affine_grid, 
    #     args=(theta, feat, False),
    #     f="custom_affine_grid.onnx",
    #     opset_version=13, verbose=True, do_constant_folding=True)


if __name__ == '__main__':
    test()