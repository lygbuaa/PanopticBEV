#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch

def fake_warp_perspective(src: torch.Tensor, M: torch.Tensor, dsize: torch.Tensor):
    N, C, _, _ = src.shape
    H, W = dsize[0], dsize[1]
    fake_output = torch.rand(N, C, H, W, dtype=torch.float, device=src.device)
    return fake_output

def fake_affine_grid(theta:torch.Tensor, feat:torch.Tensor):
    N, C, H, W = feat.size()
    fake_output = torch.rand(N, H, W, 2, dtype=torch.float, device=feat.device)
    return fake_output

def fake_deg2rad(input:torch.Tensor):
    return torch.div(input, 57.3)

def fake_rot90(input:torch.Tensor, k:int, dims:list):
    N, C, Hin, Win = input.shape
    if k == 1:
        Hout = Win
        Wout = Hin
    else: #k=2
        Hout = Hin
        Wout = Win
    fake_output = torch.rand(N, C, Hout, Wout, dtype=torch.float, device=input.device)
    return fake_output

def fake_grid_sample(input:torch.Tensor, grid:torch.Tensor):
    N, C, Hin, Win = input.shape
    _, Hout, Wout, _ = grid.shape
    fake_output = torch.rand(N, C, Hout, Wout, dtype=torch.float, device=input.device)
    return fake_output

def fake_linalg_inv(input:torch.Tensor):
    # output = torch.linalg.inv(input)
    return input

if __name__ == '__main__':
    input = torch.rand(1, 3, 480, 640, dtype=torch.float)
    grid = torch.rand(1, 96, 113, 2, dtype=torch.float)
    # output = fake_grid_sample(input, grid)
    # output = fake_linalg_inv(input)
    output = fake_rot90(input=input, k=1, dims=[2,3])
    print("output-1: {}".format(output.shape))
    output = fake_rot90(input=input, k=2, dims=[2,3])
    print("output-2: {}".format(output.shape))
    output = torch.rot90(input=input, k=1, dims=[2,3])
    print("output-3: {}".format(output.shape))
    output = torch.rot90(input=input, k=2, dims=[2,3])
    print("output-4: {}".format(output.shape))
