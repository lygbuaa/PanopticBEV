#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch
from typing import Tuple, List
# from torch.linalg import det

@torch.jit.script
def cof1(M:torch.Tensor, index:List[int]):
    zs = M[:index[0]-1, :index[1]-1]
    ys = M[:index[0]-1, index[1]:]
    zx = M[index[0]:, :index[1]-1]
    yx = M[index[0]:, index[1]:]
    # print("cof1, zs: {}, ys: {}, zx: {}, yx: {}".format(zs.shape, ys.shape, zx.shape, yx.shape))
    s = torch.cat((zs, ys), dim=1)
    x = torch.cat((zx, yx), dim=1)
    sx = torch.cat((s, x), dim=0)
    # print("cof1, s: {}, x: {}, sx: {}".format(s.shape, x.shape, sx.shape))
    return torch.linalg.det(sx)
 
@torch.jit.script
def alcof(M:torch.Tensor, index:List[int]):
    return torch.pow(-1, index[0]+index[1]) * cof1(M, index)
 
@torch.jit.script
def adj(M:torch.Tensor):
    result = torch.zeros(size=(M.shape[0], M.shape[1]), device=M.device)
    for i in range(1, M.shape[0]+1):
        for j in range(1, M.shape[1]+1):
            result[j-1][i-1] = alcof(M, [i,j])
    return result

@torch.jit.script
def custom_inverse_dim2(input:torch.Tensor):
    assert input.dim()==2, "inverse only support dims=2 tensor!"
    assert input.size(0)==input.size(1), "inverse only support square matrix!"
    output = 1.0 / torch.linalg.det(input) * adj(input)
    return output

@torch.jit.script
def custom_inverse(input:torch.Tensor) -> torch.Tensor:
    # to avoid Sequence ops, only support dim=3 && C=1
    dims = input.dim()
    assert dims == 3
    C, H, W = input.shape
    assert C == 1 and H == W
    # print("custom_inverse: input: {}".format(input.shape))
    output = custom_inverse_dim2(input[0])
    return output.unsqueeze(0)
    # assert dims <= 3, "invalid tensor dim: {}!".format(dims)

    # if dims == 1:
    #     return 1.0/input
    # elif dims == 2:
    #     return custom_inverse_dim2(input)
    # elif dims == 3:
    #     inv_list = []
    #     for A in input:
    #         inv_A = custom_inverse_dim2(A)
    #         inv_list.append(inv_A)
    #     return torch.stack(inv_list, dim=0)
    # else:
    #     # just return a tensor with the same dim
    #     return input

def test():
    x = torch.rand(size=[2, 4, 4], dtype=torch.float)
    print("original x: {}\n{}".format(x.shape, x))
    x1 = torch.inverse(input=x)
    print("torch.inverse: {}\n{}, \nproduct:\n{}".format(x1.shape, x1, x@x1))
    x2 = custom_inverse(input=x)
    diff = x2 - x1
    print("custom_inverse: {}\n{}\ndiff: \n{}, \nproduct:\n{}".format(x2.shape, x2, diff, x@x2))


if __name__ == '__main__':
    test()