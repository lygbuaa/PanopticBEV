#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch, torchvision
from typing import Tuple, List

# use torch.transopose instead of torch.rot90
@torch.jit.script
def custom_rot90_once(input:torch.Tensor, dims:List[int]):
    dim0 = int(dims[0])
    dim1 = int(dims[1])
    output = input.transpose(dim0, dim1)
    output = torch.flip(output, dims=[dim0])
    return output

@torch.jit.script
def custom_rot90(input:torch.Tensor, k:int, dims:List[int]):
    assert k > 0, 'k < 0 not considered!'
    assert k < 3, 'only k=1, k=2 supported!'
    if k == 1:
        return custom_rot90_once(input, dims)
    else:
        tmp = custom_rot90_once(input, dims)
        return custom_rot90_once(tmp, dims)
        
    # output = input.new_zeros(size=input.shape)
    # for idx in range(k):
    #     output = custom_rot90_once(input, dims)
    #     input = output
    # return output

def test():
    x = torch.rand(size=[1, 2, 3, 4], dtype=torch.float)
    k = 2
    dims = [2, 1]
    # x = torch.rand(size=[3, 4], dtype=torch.float)
    print("original x: {}\n{}".format(x.shape, x))
    x1 = torch.rot90(input=x, k=k, dims=dims)
    print("torch.rot90: {}\n{}".format(x1.shape, x1))
    x2 = custom_rot90(input=x, k=k, dims=dims)
    diff = x2 - x1
    print("custom_rot90: {}\n{}\ndiff: \n{}".format(x2.shape, x2, diff))

    # custom_rot90_ts = torch.jit.trace(custom_rot90, (x, k, dims), check_trace=True)

    torch.onnx.export(
        model=custom_rot90, 
        args=(x, k, dims),
        f="custom_rot90.onnx",
        custom_opsets={"custom_domain": 1},
        opset_version=13, verbose=True, do_constant_folding=True)

if __name__ == '__main__':
    test()