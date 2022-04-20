#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, sys, time
import numpy as np
import torch
import torchvision
import onnx
import onnxruntime as ort
import onnxsim

class DemoModule(torch.nn.Module):
    def __init__(self):
        super(DemoModule, self).__init__()

    def forward(self):
        N, C, H, W = 4, 2, 3, 3
        rois = torch.rand([N, C, H, W], dtype=torch.float)
        target = (N*torch.rand([N], dtype=torch.float)).type(torch.long)
        print("orignal rois: {}".format(rois.shape))

        final_rois = torch.empty([0, C, H, W])
        for idx in range(N):
            msk = target == idx
            msk_uni, msk_cnts = torch.unique(input=msk.type(torch.long), return_counts=True)
            print("###  msk: {}-{}-{}".format(msk.shape, msk_uni, msk_cnts))
            if msk.any():
                tcnt = int(msk_cnts[1])
                new_rois = torch.zeros(size=[tcnt, C, H, W])
                rois_i = rois[msk]
                print("#-{}-# rois_i: {}, new_rois: {}".format(idx, rois_i.shape, new_rois.shape))

                ### onnxruntime can't handle this operation: 
                # RuntimeError: input_shape_value == reshape_value || input_shape_value == 1 || reshape_value == 1INTERNAL 
                # ONNX Expand input shape constraint not satisfied.
                # rois[msk] = new_rois

                final_rois = torch.cat((final_rois, rois[msk]), dim=0)
        print("final rois: {}".format(rois.shape))
        return final_rois

def onnx_export():
    mdemo = DemoModule()
    mdemo()
    demo_jit = torch.jit.script(mdemo)
    # print(demo_jit.graph)
    torch.onnx.export(
        model=demo_jit, 
        args=(),
        f="./demo_func.onnx",
        opset_version=13, verbose=False, do_constant_folding=False
    )

def run_onnx_model(model_path="./demo_func.onnx"):
    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ortss = ort.InferenceSession(model_path, providers=provider)
    print("load model {} onto {}".format(model_path, ortss.get_providers()))

    inputs=[]

    input_layers = ortss.get_inputs()
    output_layers = ortss.get_outputs()
    input_dict = {}
    output_list = []
    for idx, in_layer in enumerate(input_layers):
        input_dict[in_layer.name] = inputs[idx]
    print("input_dict: {}".format(input_dict))

    for idx, out_layer in enumerate(output_layers):
        output_list.append(out_layer.name)

    outputs = ortss.run(output_list, input_dict)
    print("outputs: {}".format(outputs))

if __name__ == "__main__":
    print("onnx version: {}, onnxruntime version: {}, device: {}".format(onnx.__version__, ort.__version__, ort.get_device()))
    onnx_export()
    run_onnx_model()