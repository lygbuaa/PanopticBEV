import os, sys, time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference
from panoptic_bev.utils import plogging
global logger


def print_version():
    # print("torch: {}".format(torch.__version__))
    logger.info("onnx version: {}".format(onnx.__version__))
    logger.info("onnxruntime version: {}".format(ort.__version__))

def load_onnx_model(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    try:
        onnx.checker.check_model(onnx_model)
        # logger.info(onnx.helper.printable_graph(onnx_model.graph))
        logger.info('onnx model graph is:\n{}'.format(onnx_model))
    except Exception as e:
        logger.error("onnx check model error: {}".format(e))
    return onnx_model

def save_onnx_model(onnx_model, onnx_model_path):
    onnx.save(onnx_model, onnx_model_path)

def modify_opset_version(onnx_model, new_version=16):
    onnx_model.opset_import[0].version = new_version
    return onnx_model

def remove_custom_domain(onnx_model, op_type="GridSample"):
    counter = 0
    for node in onnx_model.graph.node:
        # logger.info("pick graph node: {}".format(node))
        if node.op_type == 'GridSample':
            counter += 1
            node.ClearField("domain")
            logger.info("[{}] find GridSmple: {}".format(counter, node.name))
            for input_name in node.input:
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == input_name:
                        logger.info("[{}] GridSmple input_{}: {}".format(counter, input_name, value_info))
            for output_name in node.output:
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == output_name:
                        logger.info("[{}] GridSmple output_{}: {}".format(counter, output_name, value_info))
    return onnx_model

def ort_load_onnx(onnx_model_path):
    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ortss = ort.InferenceSession(onnx_model_path, providers=provider)
    logger.info("load model {} onto {}".format(onnx_model_path, ortss.get_providers()))
    return ortss

#ortss: onnxruntime loaded model, inputs: [input0, input1]
def run_onnx_model(ortss, inputs):
    input_layers = ortss.get_inputs()
    output_layers = ortss.get_outputs()
    logger.debug("input_layers: {}, output_layers: {}".format(input_layers, output_layers))
    input_dict = {}
    output_list = []
    for idx, in_layer in enumerate(input_layers):
        input_dict[in_layer.name] = inputs[idx]
        logger.debug("[{}]- in_layer: {}".format(idx, in_layer))
    # logger.debug("input_dict: {}".format(input_dict))

    for idx, out_layer in enumerate(output_layers):
        output_list.append(out_layer.name)
        # logger.debug("[{}]- out_layer: {}".format(idx, out_layer))
    # logger.debug("output_list: {}".format(output_list))

    outputs = ortss.run(output_list, input_dict)
    # logger.info("outputs: {}".format(outputs))
    return outputs

def test_transformer(onnx_model_path):
    ortss = ort_load_onnx(onnx_model_path)
    ms_feat_0 = np.random.rand(1, 160, 112, 192).astype(np.float32)
    ms_feat_1 = np.random.rand(1, 160, 56, 96).astype(np.float32)
    ms_feat_2 = np.random.rand(1, 160, 28, 48).astype(np.float32)
    ms_feat_3 = np.random.rand(1, 160, 14, 24).astype(np.float32)
    intrin = np.random.rand(1, 3, 3).astype(np.float32)
    # extrin = np.zeros((2, 3)).astype(np.float32)
    extrin = np.random.rand(2, 3).astype(np.float32)
    msk = np.random.rand(896, 768).astype(np.float32)
    inputs=[ms_feat_0, ms_feat_1, ms_feat_2, ms_feat_3, intrin, extrin, msk]
    result = run_onnx_model(ortss, inputs)
    for output in result:
        logger.info("output: {}".format(output.shape))

if __name__ == "__main__":
    plogging.init("./", "onnx_modify")
    logger = plogging.get_logger()
    print_version()
    transformer_op13_path = "../../onnx/transformer_op13.onnx"
    transformer_op16_path = "../../onnx/transformer_op16.onnx"

    # model_op13 = load_onnx_model(transformer_op13_path)
    # model_op16 = modify_opset_version(model_op13, 16)
    # model_op16 = remove_custom_domain(model_op16)
    # save_onnx_model(model_op16, transformer_op16_path)
    # ort_load_onnx(transformer_op16_path)
    test_transformer(transformer_op16_path)
