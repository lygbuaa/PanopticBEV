#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, sys, time
import numpy as np
import torch
import torchvision
import torch_tensorrt


def print_version():
    print("torch: {}".format(torch.__version__))
    print("torch_tensorrt: {}".format(torch_tensorrt.__version__))

# def benchmark(model, input_shape=(1, 3, 224, 224), dtype=torch.float32, nwarmup=10, nruns=10000):
def benchmark(model, input_data, nwarmup=10, nruns=10000):
    model = model.to(input_data.device)

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

def make_calibrator():
    cache_file_path = "./resnet.calibration.cifar10"
    if os.path.isfile(cache_file_path):
        print("find calibration cache: {}".format(cache_file_path))
        calibrator = torch_tensorrt.ptq.CacheCalibrator(cache_file_path)
    else:
        print("create calibration cache: {}".format(cache_file_path))
        testing_dataset = torchvision.datasets.CIFAR10(root='/home/hugoliu/github/dataset/cifar/cifar-10-python',
                                                            train=False,
                                                            download=True,
                                                            transform=torchvision.transforms.Compose([
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010))
                                                            ]))

        testing_dataloader = torch.utils.data.DataLoader(testing_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=1)

        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(testing_dataloader,
                                                    cache_file=cache_file_path,
                                                    use_cache=False,
                                                    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                                                    device=torch.device('cuda:0'))
    return calibrator

def test_resnet_trt():
    device=torch.device('cuda:0')
    # device=torch.device('cpu')
    input_fp32 = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    input_fp16 = torch.randn((1, 3, 224, 224), dtype=torch.float16, device=device)
    input_int8 = torch.randint(low=0, high=255, size=(1, 3, 224, 224), dtype=torch.uint8, device=device)

    model = torchvision.models.resnet50(pretrained=True).eval()
    print("###### benchmark torchvision.models.resnet50 ######")
    benchmark(model, input_fp32, nwarmup=100, nruns=100)

    print("###### benchmark torch.jit.resnet50 ######")
    script_model = torch.jit.script(model)
    benchmark(script_model, input_fp32, nwarmup=100, nruns=100)
    torch.jit.save(script_model, "./resnet50_fp32.jit")

    # print("###### benchmark jit.optimized.resnet50 ######")
    # frozen_model = torch.jit.optimize_for_inference(script_model)
    # benchmark(frozen_model, input_fp32, nwarmup=100, nruns=100)

    print("###### benchmark trt.fp32.resnet50 ######")
    trt_model_fp32 = torch_tensorrt.compile(
        module=script_model, 
        inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float32)], 
        enabled_precisions={torch.float32},
        workspace_size=1<<22
    )
    benchmark(trt_model_fp32, input_fp32, nwarmup=100, nruns=100)
    torch.jit.save(script_model, "./resnet50_fp32.trt")

    print("###### benchmark trt.fp16.resnet50 ######")
    trt_model_fp16 = torch_tensorrt.compile(
        module=script_model, 
        inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float16)], 
        enabled_precisions={torch.float32, torch.float16},
        workspace_size=1<<22
    )
    benchmark(trt_model_fp16, input_fp16, nwarmup=100, nruns=100)
    torch.jit.save(script_model, "./resnet50_fp16.jit")

    print("###### benchmark trt.int8.resnet50 ######")
    calibrator = make_calibrator()
    trt_model_int8 = torch_tensorrt.compile(
        module=script_model,
        inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float)],
        enabled_precisions={torch.int8},
        calibrator=calibrator,
        workspace_size=1<<32
    )
    benchmark(trt_model_int8, input_fp32, nwarmup=100, nruns=100)
    # support torch.jit.save() & torch.jit.load()
    torch.jit.save(trt_model_int8, "./resnet50_int8.trt")


def test_resnet_int8():
    device=torch.device('cuda:0')
    # device=torch.device('cpu')
    input_fp32 = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    input_fp16 = torch.randn((1, 3, 224, 224), dtype=torch.float16, device=device)
    input_int8 = torch.randint(low=-128, high=127, size=(1, 3, 224, 224), dtype=torch.int8, device=device)
    model = torchvision.models.resnet50(pretrained=True).eval()
    print("###### benchmark torchvision.models.resnet50 ######")
    benchmark(model, input_fp32, nwarmup=100, nruns=100)

    script_model = torch.jit.script(model)
    torch.jit.save(script_model, "./resnet50_fp32.jit")
    calibrator = make_calibrator()
    trt_model_int8 = torch_tensorrt.compile(
        module=script_model,
        inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float)],
        enabled_precisions={torch.int8},
        calibrator=calibrator,
        workspace_size=1<<32
    )

    print("###### benchmark trt.model.int8 ######")
    benchmark(trt_model_int8, input_fp32, nwarmup=100, nruns=100)
    # support torch.jit.save() & torch.jit.load()
    torch.jit.save(trt_model_int8, "./resnet50_int8.trt")


if __name__ == '__main__':
    print_version()
    test_resnet_trt()