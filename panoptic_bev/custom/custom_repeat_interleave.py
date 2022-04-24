#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch, torchvision

def custom_repeat_interleave(input:torch.Tensor, repeats:int, dim:int=0):
    C, H, W = input.shape
    ss = []
    if dim == 0:
        for i in range(C):
            s = input[i, :, :]
            ss.append(s.expand(repeats, H, W))
    elif dim == 1:
        for i in range(H):
            s = input[:, i, :].view(C, 1, W)
            ss.append(s.expand(C, repeats, W))
    elif dim == 2:
        for i in range(W):
            s = input[:, :, i].view(C, H, 1)
            ss.append(s.expand(C, H, repeats))
    output = torch.cat(ss, dim=dim)
    return output

def test():
    x = torch.rand(size=[3, 4, 4], dtype=torch.float)
    print("original x: {}".format(x))
    x1 = torch.repeat_interleave(input=x, repeats=2, dim=0)
    print("torch.repeat_interleave: {}".format(x1))
    x2 = custom_repeat_interleave(input=x, repeats=2, dim=0)
    print("custom_repeat_interleave: {}".format(x2))

if __name__ == '__main__':
    test()