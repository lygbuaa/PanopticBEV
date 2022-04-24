#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch, torchvision

def fake_target_level(boxes:torch.Tensor):
    N, _ = boxes.shape
    target_level = torch.rand([N], dtype = torch.float, device=boxes.device)
    return target_level

def fake_prediction_generator(boxes:torch.Tensor, scores:torch.Tensor, nms_threshold:float=0.3, score_threshold:float=0.1, max_predictions:int=100):
    bbx_pred = torch.rand([2, 4], dtype=torch.float, device=boxes.device)
    cls_pred = torch.tensor([6, 7], dtype = torch.long, device=boxes.device)
    obj_pred = torch.rand([2], dtype = torch.float, device=boxes.device)
    return torch.unsqueeze(bbx_pred, dim=0), torch.unsqueeze(cls_pred, dim=0), torch.unsqueeze(obj_pred, dim=0)

def fake_shift_boxes(bbx:torch.Tensor, shift:torch.Tensor, dim:int=-1):
    N, _, _ = bbx.shape
    boxes = torch.rand(N, 4, 4, dtype=torch.float, device=bbx.device)
    return boxes

def fake_head_roi_msk(rois:torch.Tensor):
    N, _, _, _ = rois.shape
    msk_logits = torch.rand(N, 4, 28, 28, dtype=torch.float, device=rois.device)
    return msk_logits

def fake_head_roi_bbx(rois:torch.Tensor):
    N, _, _, _ = rois.shape
    cls_logits = torch.rand(N, 5, dtype=torch.float, device=rois.device)
    bbx_logits = torch.rand(N, 4, 4, dtype=torch.float, device=rois.device)
    return cls_logits, bbx_logits

def fake_po_roi(x: torch.Tensor, bbx: torch.Tensor, idx: torch.Tensor, out_size: torch.Tensor):
    _, C, _, _ = x.shape
    N, _ = bbx.shape
    H = out_size[0]
    W = out_size[1]
    rois = torch.rand(N, C, H, W, dtype=torch.float, device=x.device)
    return rois

def torchvision_nms(bbx:torch.Tensor, scores:torch.Tensor, threshold:float, n_max:int):
    idx = torchvision.ops.nms(bbx, scores, threshold)
    return idx

def fake_idx(target_level:torch.Tensor, idx_level:int):
    idx = torch.rand(16, dtype=torch.float, device=target_level.device)*16
    # idx = torch.tensor([0], device=target_level.device).expand(target_level.size(dim=0))
    return idx.type(torch.long)
    # tmp = torch.tensor([idx_level], dtype=torch.long, device=target_level.device)#.expand(target_level.size(dim=0))
    # return target_level == tmp
    # return tmp
    # return target_level.eq(tmp)
    # idx = torch.tensor([True], device=target_level.device).expand(target_level.size(dim=0))
    # return idx

def fake_rois(proposals):
    N, _ = proposals.shape
    return torch.tensor([N, 256, 14, 14], dtype=torch.float, device=proposals.device)

def fake_po_nms(bbx:torch.Tensor, scores:torch.Tensor, threshold:float, n_max:int):
    idx = torch.rand(300, dtype=torch.float, device=bbx.device)*n_max
    return idx.type(torch.long)

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
    # output = torch.inverse(input)
    return input

def fake_repeat_interleave(input:torch.Tensor, repeats:int, dim:int=0):
    N, H, W = input.shape
    output = torch.rand(repeats, H, W, dtype=input.dtype, device=input.device)
    return output

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
