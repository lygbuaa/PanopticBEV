import os, sys, time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference
from grid_sample_value_info import g_grid_sample_table
from panoptic_bev.utils import plogging
global logger

def print_version():
    # print("torch: {}".format(torch.__version__))
    logger.info("onnx version: {}".format(onnx.__version__))
    logger.info("onnxruntime version: {}".format(ort.__version__))

def load_onnx_model(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    try:
        # logger.info('onnx model graph is:\n{}'.format(onnx_model))
        # logger.info(onnx.helper.printable_graph(onnx_model.graph))
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        logger.error("onnx check model error: {}".format(e))
    return onnx_model

def save_onnx_model(onnx_model, onnx_model_path):
    onnx.save(onnx_model, onnx_model_path)

def modify_opset_version(onnx_model, new_version=16):
    onnx_model.opset_import[0].version = new_version
    return onnx_model

def onnx_shape_infer(onnx_model):
    inferred_model = shape_inference.infer_shapes(onnx_model)
    return inferred_model

def iterate_op(onnx_model, op_type="GridSample"):
    counter = 0
    for node in onnx_model.graph.node:
        # logger.info("pick graph node: {}".format(node))
        if node.op_type == op_type:
            counter += 1
            logger.info("[{}] find {}: {}".format(counter, op_type, node.name))
            for input_name in node.input:
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == input_name:
                        logger.info("[{}] {} input_{}: {}".format(counter, op_type, input_name, value_info))
                for init in onnx_model.graph.initializer:
                    if init.name == input_name:
                        logger.info("[{}] {} input_{} is initializer {}".format(counter, op_type, input_name, init.dims))
            for output_name in node.output:
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == output_name:
                        logger.info("[{}] {} output_{}: {}".format(counter, op_type, output_name, value_info))
    return onnx_model

def add_value_info(onnx_model, name="tensor_name", elem_type=1, shape=[1, 3, 17, 17]):
    new_value_info = onnx.helper.make_tensor_value_info(name=name, elem_type=elem_type, shape=shape)
    onnx_model.graph.value_info.append(new_value_info)
    logger.info("add value_info: {}".format(new_value_info))

def fix_value_info(value_info, elem_type=1, shape=[1, 3, 17, 17]):
    #if elem_type is empty
    if not value_info.type.tensor_type.HasField("elem_type"):
        value_info.type.tensor_type.elem_type = elem_type
        logger.info("value_info fix elem_type: {}".format(value_info))
    #if shape is empty
    if not value_info.type.tensor_type.HasField("shape"):
        # onnx.helper.make_tensor_type_proto
        value_info.type.tensor_type.shape.dim.extend([])
        for d in shape:
            dim = value_info.type.tensor_type.shape.dim.add()
            dim.dim_value = d
        logger.info("value_info fix shape: {}".format(value_info))
    #if shape has dim_param
    for idx in range(len(shape)):
        if value_info.type.tensor_type.shape.dim[idx].HasField("dim_param"):
            value_info.type.tensor_type.shape.dim[idx].ClearField("dim_param")
            value_info.type.tensor_type.shape.dim[idx].dim_value = shape[idx]
            logger.info("value_info fix [{}] dim_param: {}".format(idx, value_info))

def fill_op_value_info(onnx_model, op_type="GridSample", value_info_table={}, initializer2tensor=False):
    counter = 0
    helper_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == 'GridSample':
            counter += 1
            node.ClearField("domain")
            logger.info("[{}] find {}: {}".format(counter, op_type, node.name))
            for input_name in node.input:
                found_value_info = False
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == input_name:
                        logger.info("[{}] GridSample input_{}: \n{}".format(counter, input_name, value_info))
                        found_value_info = True
                        fix_value_info(value_info, elem_type=value_info_table[input_name]["elem_type"], shape=value_info_table[input_name]["shape"])
                #should add a node here, to ensure "TRT_PluginV2 input.is_tensor()"
                for init in onnx_model.graph.initializer:
                    if init.name == input_name:
                        logger.info("[{}] {} input_{} is initializer {}".format(counter, op_type, input_name, init.dims))
                        if initializer2tensor:
                            init.name = input_name + "_weight"
                            helper_node = onnx.helper.make_node(op_type="Cast", inputs=[init.name], outputs=[input_name], name=input_name+"_Cast", to=1)
                            helper_nodes.append(helper_node)
                if not found_value_info:
                    add_value_info(onnx_model, name=input_name, elem_type=value_info_table[input_name]["elem_type"], shape=value_info_table[input_name]["shape"])
                    found_value_info = False
            for output_name in node.output:
                found_value_info = False
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == output_name:
                        logger.info("[{}] GridSample output_{}: \n{}".format(counter, output_name, value_info))
                        found_value_info = True
                        fix_value_info(value_info, elem_type=value_info_table[output_name]["elem_type"], shape=value_info_table[output_name]["shape"])
                if not found_value_info:
                    add_value_info(onnx_model, name=output_name, elem_type=value_info_table[output_name]["elem_type"], shape=value_info_table[output_name]["shape"])
                    found_value_info = False

    for node in helper_nodes:
        onnx_model.graph.node.append(node)
        logger.info("append helper node: {}".format(node))

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

# polygraphy surgeon sanitize transformer_op13.onnx --fold-constants --output transformer_op13_folded.onnx
def modify_transformer_step1(input_onnx_path, output_onnx_path):
    model = load_onnx_model(input_onnx_path)
    model = fill_op_value_info(model, op_type="GridSample", value_info_table=g_grid_sample_table, initializer2tensor=True)
    model = modify_opset_version(model, new_version=16)
    model = onnx_shape_infer(model)
    model = modify_opset_version(model, new_version=13)
    save_onnx_model(model, output_onnx_path)

def modify_transformer_step2(input_onnx_path, output_onnx_path):
    model = load_onnx_model(input_onnx_path)
    model = fill_op_value_info(model, op_type="GridSample", value_info_table=g_grid_sample_table, initializer2tensor=True)
    model = modify_opset_version(model, new_version=13)
    save_onnx_model(model, output_onnx_path)

if __name__ == "__main__":
    plogging.init("./", "onnx_modify")
    logger = plogging.get_logger()
    print_version()
    input_onnx_path = "../../onnx/transformer_op13_folded.onnx"
    # input_onnx_path = "../../onnx/vit_op16_folded.onnx"
    # output_onnx_path = "../../onnx/vit_op16.onnx"
    output_onnx_path = "../../onnx/vit_op13_folded.onnx"

    # modify_transformer_step1(input_onnx_path, output_onnx_path)
    onnx_model = onnx.load("../../onnx/transformer_op13.onnx")
    logger.info('onnx model graph is:\n{}'.format(onnx_model))

    # model_op13 = load_onnx_model(input_onnx_path)
    # model_op13 = fill_op_value_info(model_op13, op_type="GridSample", value_info_table=g_grid_sample_table)
    # model_op13 = onnx_shape_infer(model_op13)
    # save_onnx_model(model_op13, output_onnx_path)

    # iterate_op(model_op13)

    # model_op16 = load_onnx_model(output_onnx_path)
    # ort_load_onnx(transformer_op16_path)
    # test_transformer(transformer_op13_path)
