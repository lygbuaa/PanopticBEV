#! /bin/bash
# set -ex

# pip install pycuda
# pip install onnxruntime-gpu
# pip install onnx-graphsurgeon
reset
# LD_PRELOAD=./TensorRT/build/out/libnvinfer_plugin_static.a python test_plugin_result.py
export LD_PRELOAD=/home/hugoliu/github/onnxparser-trt-plugin-sample/TensorRT/build/out/libnvinfer_plugin.so
# polygraphy inspect model ../transformer_fp16.trt --model-type=engine
# polygraphy inspect model ../backbone_fp16.trt --model-type=engine
python onnx2trt_test.py
unset LD_PRELOAD

