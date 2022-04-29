#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch
from typing import Tuple, List
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from panoptic_bev.utils.roi_sampling import roi_sampling

# torch.ops.load_library("/home/hugoliu/github/PanopticBEV/panoptic_bev/utils/roi_sampling/_backend.cpython-38-x86_64-linux-gnu.so")

# Define custom symbolic function
@parse_args("v", "v", "v", "v")
def symbolic_po_roi(g:torch._C.Graph, x:torch._C.Value, proposals:torch._C.Value, proposals_idx:torch._C.Value, roi_size:torch._C.Value)->torch._C.Value:
    return g.op("custom_domain::po_roi", x, proposals, proposals_idx, roi_size)

# Register custom symbolic function  
register_custom_op_symbolic("po_cpp_ops::po_roi", symbolic_po_roi, 1)

@torch.jit.script
def custom_po_roi(x:torch.Tensor, proposals:torch.Tensor, proposals_idx:torch.Tensor, roi_size:torch.Tensor):
    rois = torch.ops.po_cpp_ops.po_roi(x, proposals, proposals_idx, roi_size)
    return rois

def test():
    x = torch.rand(size=[1, 256, 112, 192], dtype=torch.float)
    proposals = torch.rand(size=[7, 4], dtype=torch.float)
    proposals_idx = torch.tensor([1, 2, 1, 1, 2, 2, 1], dtype=torch.long)
    roi_size = torch.tensor([14, 14], dtype=torch.int)
    rois = custom_po_roi(x, proposals, proposals_idx, roi_size)
    print("rois: {}".format(rois.shape))

    torch.onnx.export(
        model=custom_po_roi, 
        args=(x, proposals, proposals_idx, roi_size),
        f="custom_po_roi.onnx",
        custom_opsets={"custom_domain": 1},
        opset_version=13, verbose=True, do_constant_folding=True)

if __name__ == '__main__':
    test()