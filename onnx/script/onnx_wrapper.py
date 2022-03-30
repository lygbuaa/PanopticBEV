#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, sys, time
import numpy as np
import torch
import torchvision
# import torch_tensorrt
import onnx
import onnxruntime as ort
import onnxsim
from panoptic_bev.utils import plogging
# plogging.init("./", "onnx_wrapper")
logger = plogging.get_logger()

class OnnxWrapper(object):
    def __init__(self):
        self.print_version()

    def __init__(self, model_path):
        self.ortss = self.load_onnx_model(model_path)

    def print_version(self):
        # print("torch: {}".format(torch.__version__))
        logger.info("onnx version: {}".format(onnx.__version__))
        logger.info("onnx runtime version: {}".format(ort.__version__))
        logger.info("onnx runtime: {}".format(ort.get_device()))

    def benchmark(self, ortss, inputs, nwarmup=10, nruns=10000):
        logger.debug("Warm up ...")
        with torch.no_grad():
            for _ in range(nwarmup):
                features = self.run_onnx_model(ortss, inputs)
        
        logger.debug("Start timing ...")
        timings = []
        for i in range(1, nruns+1):
            start_time = time.time()
            features = self.run_onnx_model(ortss, inputs)
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                logger.debug('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
        logger.debug('Average batch time: %.2f ms'%(np.mean(timings)*1000))

    def load_onnx_model(self, model_path):
        provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ortss = ort.InferenceSession(model_path, providers=provider)
        logger.info("load model {} onto {}".format(model_path, ortss.get_providers()))
        return ortss

    #ortss: onnxruntime loaded model, inputs: [input0, input1]
    def run_onnx_model(self, ortss, inputs):
        input_layers = ortss.get_inputs()
        output_layers = ortss.get_outputs()
        input_dict = {}
        output_list = []
        for idx, in_layer in enumerate(input_layers):
            input_dict[in_layer.name] = inputs[idx]
            # logger.debug("[{}]- in_layer: {}".format(idx, in_layer))
        # logger.debug("input_dict: {}".format(input_dict))

        for idx, out_layer in enumerate(output_layers):
            output_list.append(out_layer.name)
            # logger.debug("[{}]- out_layer: {}".format(idx, out_layer))
        # logger.debug("output_list: {}".format(output_list))

        outputs = ortss.run(output_list, input_dict)
        # logger.info("outputs: {}".format(outputs))
        return outputs

    def simplify_onnx_model(self, model_path):
        model = onnx.load(model_path)
        logger.debug("simplify onnx model: {}".format(model_path))
        # logger.info('onnx model graph is:\n{}'.format(model.graph))
        model_sim, check = onnxsim.simplify(model)
        logger.debug("onnxsim check: {}".format(check))
        new_path = model_path + ".sim"
        onnx.save(model_sim, new_path)
        logger.debug("simplify model saved to: {}".format(new_path))

    # run onnx with torch.Tensors
    def run(self, input_tensor_list):
        input_np_list = []
        device = torch.device('cuda:0')
        for idx, input_tensor in enumerate(input_tensor_list):
            device = input_tensor.device
            input_np = input_tensor.cpu().numpy()
            logger.debug("input-[{}]: {}".format(idx, input_np.shape))
            input_np_list.append(input_np)

        output_np_list = self.run_onnx_model(self.ortss, input_np_list)
        output_tensor_list = []
        for idx, output_np in enumerate(output_np_list):
            output_tensor = torch.tensor(output_np, device=device)
            output_tensor_list.append(output_tensor)
        return output_tensor_list

    def test_encoder(self, model_path):
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            logger.error("onnx check model error: {}".format(e))
            return False

        ortss = self.load_onnx_model(model_path)
        image = np.random.rand(1, 3, 448, 768).astype(np.float32)
        self.benchmark(ortss, [image], nwarmup=100, nruns=100)
        # result = self.run_onnx_model(ortss, [image])
        # logger.info(result)
        return True

    def test_resnet50(self, model_path):
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        # logger.info(onnx.helper.printable_graph(model.graph))
        ortss = self.load_onnx_model(model_path)
        image = np.random.rand(1, 3, 224, 224).astype(np.float32)
        result = self.run_onnx_model(ortss, [image])

    def test_torch(self):
        image_torch = torch.rand(1, 3, 224, 224, dtype=torch.float)
        result_torch = self.run([image_torch])
        print(result_torch)

if __name__ == "__main__":
    resnet50_path = "./resnet50-v1-12/resnet50-v1-12.onnx"
    encoder_path = "../body_encoder_op13.onnx"
    encoder_sim_path = "../body_encoder_op13_sim.onnx"
    onwp = OnnxWrapper(resnet50_path)
    onwp.test_torch()
    # onwp.simplify_onnx_model(encoder_path)
    # onwp.test_encoder(encoder_sim_path)

