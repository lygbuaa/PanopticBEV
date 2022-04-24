#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch, torchvision

# use torch.transopose instead of torch.rot90
def custom_rot90_once(input:torch.Tensor, dims:list):
    dim0 = dims[0]
    dim1 = dims[1]
    output = input.transpose(dim0, dim1)
    output = torch.flip(output, dims=[dim0])
    return output

def custom_rot90(input:torch.Tensor, k:int, dims:list):
    assert k > 0, 'k < 0 not considered!'
    output = None
    for idx in range(k):
        output = custom_rot90_once(input, dims)
        input = output
    return output

def test():
    x = torch.rand(size=[1, 2, 3, 4], dtype=torch.float)
    # x = torch.rand(size=[3, 4], dtype=torch.float)
    print("original x: {}\n{}".format(x.shape, x))
    x1 = torch.rot90(input=x, k=3, dims=[2, 1])
    print("torch.rot90: {}\n{}".format(x1.shape, x1))
    x2 = custom_rot90(input=x, k=3, dims=[2, 1])
    diff = x2 - x1
    print("custom_rot90: {}\n{}\ndiff: \n{}".format(x2.shape, x2, diff))


if __name__ == '__main__':
    test()