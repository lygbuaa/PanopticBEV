
import numpy as np
import json, onnx
from panoptic_bev.utils import plogging
global logger

'''
# original table, from model scripts
g_grid_sample_table = {
#GridSample_148
"756": {"elem_type": 1, "shape": [1, 128, 192, 192]},
"854": {"elem_type": 1, "shape": [1, 192, 225, 2]},
"855": {"elem_type": 1, "shape": [1, 128, 192, 225]},
#GridSample_224
"867": {"elem_type": 1, "shape": [1, 128, 112, 192]},
"998": {"elem_type": 1, "shape": [1, 192, 224, 2]},
"999": {"elem_type": 1, "shape": [1, 128, 192, 224]},
#GridSample_357
"1030": {"elem_type": 1, "shape": [1, 1, 192, 224]},
"1177": {"elem_type": 1, "shape": [1, 112, 192, 2]},
"1178": {"elem_type": 1, "shape": [1, 1, 112, 192]},
#GridSample_462
"1191": {"elem_type": 1, "shape": [1, 128, 224, 192]},
"1295": {"elem_type": 1, "shape": [1, 192, 225, 2]},
"1296": {"elem_type": 1, "shape": [1, 128, 192, 225]},
#GridSample_941
"1426": {"elem_type": 1, "shape": [1, 256, 224, 384]},
"1831": {"elem_type": 1, "shape": [1, 224, 384, 2]},
"1832": {"elem_type": 1, "shape": [1, 256, 224, 384]},

#GridSample_1091
"1898": {"elem_type": 1, "shape": [1, 128, 96, 96]},
"1996": {"elem_type": 1, "shape": [1, 96, 113, 2]},
"1997": {"elem_type": 1, "shape": [1, 128, 96, 113]},
#GridSample_1167
"2009": {"elem_type": 1, "shape": [1, 128, 56, 96]},
"2140": {"elem_type": 1, "shape": [1, 96, 112, 2]},
"2141": {"elem_type": 1, "shape": [1, 128, 96, 112]},
#GridSample_1300
"2172": {"elem_type": 1, "shape": [1, 1, 96, 112]},
"2319": {"elem_type": 1, "shape": [1, 56, 96, 2]},
"2320": {"elem_type": 1, "shape": [1, 1, 56, 96]},
#GridSample_1405
"2333": {"elem_type": 1, "shape": [1, 128, 112, 96]},
"2437": {"elem_type": 1, "shape": [1, 96, 113, 2]},
"2438": {"elem_type": 1, "shape": [1, 128, 96, 113]},
#GridSample_1884
"2568": {"elem_type": 1, "shape": [1, 256, 112, 192]},
"2973": {"elem_type": 1, "shape": [1, 112, 192, 2]},
"2974": {"elem_type": 1, "shape": [1, 256, 112, 192]},

#GridSample_2034
"3040": {"elem_type": 1, "shape": [1, 128, 48, 48]},
"3138": {"elem_type": 1, "shape": [1, 48, 57, 2]},
"3139": {"elem_type": 1, "shape": [1, 128, 48, 57]},
#GridSample_2109
"3151": {"elem_type": 1, "shape": [1, 128, 28, 48]},
"3292": {"elem_type": 1, "shape": [1, 48, 56, 2]},
"3293": {"elem_type": 1, "shape": [1, 128, 48, 56]},
#GridSample_2242
"3324": {"elem_type": 1, "shape": [1, 1, 48, 56]},
"3471": {"elem_type": 1, "shape": [1, 28, 48, 2]},
"3472": {"elem_type": 1, "shape": [1, 1, 28, 48]},
#GridSample_2347
"3485": {"elem_type": 1, "shape": [1, 128, 56, 48]},
"3589": {"elem_type": 1, "shape": [1, 48, 57, 2]},
"3590": {"elem_type": 1, "shape": [1, 128, 48, 57]},
#GridSample_2826
"3720": {"elem_type": 1, "shape": [1, 256, 56, 96]},
"4125": {"elem_type": 1, "shape": [1, 56, 96, 2]},
"4126": {"elem_type": 1, "shape": [1, 256, 56, 96]},

#GridSample_2976
"4192": {"elem_type": 1, "shape": [1, 128, 24, 24]},
"4290": {"elem_type": 1, "shape": [1, 24, 29, 2]},
"4291": {"elem_type": 1, "shape": [1, 128, 24, 29]},
#GridSample_3052
"4303": {"elem_type": 1, "shape": [1, 128, 14, 24]},
"4434": {"elem_type": 1, "shape": [1, 24, 28, 2]},
"4435": {"elem_type": 1, "shape": [1, 128, 24, 28]},
#GridSample_3185
"4466": {"elem_type": 1, "shape": [1, 1, 24, 28]},
"4613": {"elem_type": 1, "shape": [1, 14, 24, 2]},
"4614": {"elem_type": 1, "shape": [1, 1, 14, 24]},
#GridSample_3290
"4627": {"elem_type": 1, "shape": [1, 128, 28, 24]},
"4731": {"elem_type": 1, "shape": [1, 24, 29, 2]},
"4732": {"elem_type": 1, "shape": [1, 128, 24, 29]},
#GridSample_3769
"4862": {"elem_type": 1, "shape": [1, 256, 28, 48]},
"5267": {"elem_type": 1, "shape": [1, 28, 48, 2]},
"5268": {"elem_type": 1, "shape": [1, 256, 28, 48]},
}
'''

# refresh this table after onnx model re-export
g_grid_sample_table = {
    "754": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            192,
            192
        ]
    },
    "852": {
        "elem_type": 1,
        "shape": [
            1,
            192,
            225,
            2
        ]
    },
    "853": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            192,
            225
        ]
    },
    "865": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            112,
            192
        ]
    },
    "996": {
        "elem_type": 1,
        "shape": [
            1,
            192,
            224,
            2
        ]
    },
    "997": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            192,
            224
        ]
    },
    "1028": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            192,
            224
        ]
    },
    "1175": {
        "elem_type": 1,
        "shape": [
            1,
            112,
            192,
            2
        ]
    },
    "1176": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            112,
            192
        ]
    },
    "1189": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            224,
            192
        ]
    },
    "1293": {
        "elem_type": 1,
        "shape": [
            1,
            192,
            225,
            2
        ]
    },
    "1294": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            192,
            225
        ]
    },
    "1390": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            224,
            384
        ]
    },
    "1391": {
        "elem_type": 1,
        "shape": [
            1,
            224,
            384,
            2
        ]
    },
    "1392": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            224,
            384
        ]
    },
    "1457": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            96,
            96
        ]
    },
    "1555": {
        "elem_type": 1,
        "shape": [
            1,
            96,
            113,
            2
        ]
    },
    "1556": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            96,
            113
        ]
    },
    "1568": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            56,
            96
        ]
    },
    "1699": {
        "elem_type": 1,
        "shape": [
            1,
            96,
            112,
            2
        ]
    },
    "1700": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            96,
            112
        ]
    },
    "1731": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            96,
            112
        ]
    },
    "1878": {
        "elem_type": 1,
        "shape": [
            1,
            56,
            96,
            2
        ]
    },
    "1879": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            56,
            96
        ]
    },
    "1892": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            112,
            96
        ]
    },
    "1996": {
        "elem_type": 1,
        "shape": [
            1,
            96,
            113,
            2
        ]
    },
    "1997": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            96,
            113
        ]
    },
    "2093": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            112,
            192
        ]
    },
    "2094": {
        "elem_type": 1,
        "shape": [
            1,
            112,
            192,
            2
        ]
    },
    "2095": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            112,
            192
        ]
    },
    "2160": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            48,
            48
        ]
    },
    "2258": {
        "elem_type": 1,
        "shape": [
            1,
            48,
            57,
            2
        ]
    },
    "2259": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            48,
            57
        ]
    },
    "2271": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            28,
            48
        ]
    },
    "2402": {
        "elem_type": 1,
        "shape": [
            1,
            48,
            56,
            2
        ]
    },
    "2403": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            48,
            56
        ]
    },
    "2434": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            48,
            56
        ]
    },
    "2581": {
        "elem_type": 1,
        "shape": [
            1,
            28,
            48,
            2
        ]
    },
    "2582": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            28,
            48
        ]
    },
    "2595": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            56,
            48
        ]
    },
    "2699": {
        "elem_type": 1,
        "shape": [
            1,
            48,
            57,
            2
        ]
    },
    "2700": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            48,
            57
        ]
    },
    "2796": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            56,
            96
        ]
    },
    "2797": {
        "elem_type": 1,
        "shape": [
            1,
            56,
            96,
            2
        ]
    },
    "2798": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            56,
            96
        ]
    },
    "2863": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            24,
            24
        ]
    },
    "2961": {
        "elem_type": 1,
        "shape": [
            1,
            24,
            29,
            2
        ]
    },
    "2962": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            24,
            29
        ]
    },
    "2974": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            14,
            24
        ]
    },
    "3105": {
        "elem_type": 1,
        "shape": [
            1,
            24,
            28,
            2
        ]
    },
    "3106": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            24,
            28
        ]
    },
    "3137": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            24,
            28
        ]
    },
    "3284": {
        "elem_type": 1,
        "shape": [
            1,
            14,
            24,
            2
        ]
    },
    "3285": {
        "elem_type": 1,
        "shape": [
            1,
            1,
            14,
            24
        ]
    },
    "3298": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            28,
            24
        ]
    },
    "3402": {
        "elem_type": 1,
        "shape": [
            1,
            24,
            29,
            2
        ]
    },
    "3403": {
        "elem_type": 1,
        "shape": [
            1,
            128,
            24,
            29
        ]
    },
    "3499": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            28,
            48
        ]
    },
    "3500": {
        "elem_type": 1,
        "shape": [
            1,
            28,
            48,
            2
        ]
    },
    "3501": {
        "elem_type": 1,
        "shape": [
            1,
            256,
            28,
            48
        ]
    }
}


def print_table(table):
    logger.info("table: {}".format(json.dumps(table, sort_keys=False, indent=4)))
    # counter = 0
    # for key in table:
    #     logger.info("[{}] {}-{}".format(counter, key, table[key]))
    #     counter += 1

def get_value_with_counter(op_counter, tensor_counter):
    # 3 tensor for each op: input*2, output*1
    total_counter = op_counter*3 + tensor_counter
    counter = 0
    for key in g_grid_sample_table:
        if counter == total_counter:
            # logger.info("[{}] {}-{}".format(counter, key, g_grid_sample_table[key]))
            return key, g_grid_sample_table[key]
        counter += 1
    return None, None

def refresh_table(onnx_model, op_type="GridSample"):
    new_grid_sample_table = {}
    op_counter = 0
    for node in onnx_model.graph.node:
        # logger.info("pick graph node: {}".format(node))
        tensor_counter = 0
        if node.op_type == op_type:
            logger.info("[{}] find {}: {}".format(op_counter, op_type, node.name))
            for tensor_name in node.input:
                pre_key, pre_val = get_value_with_counter(op_counter, tensor_counter)
                logger.info("[{}-{}] tensor_{} pre key: {}, pre val: {}".format(op_counter, tensor_counter, tensor_name, pre_key, pre_val))
                new_grid_sample_table[tensor_name] = pre_val
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == tensor_name:
                        logger.info("[{}-{}] tensor_{}: has value_info {}".format(op_counter, tensor_counter, tensor_name, value_info))
                for init in onnx_model.graph.initializer:
                    if init.name == tensor_name:
                        logger.info("[{}-{}] tensor_{} is initializer {}".format(op_counter, tensor_counter, tensor_name, init.dims))
                tensor_counter += 1
            for tensor_name in node.output:
                pre_key, pre_val = get_value_with_counter(op_counter, tensor_counter)
                logger.info("[{}-{}] tensor_{} pre key: {}, pre val: {}".format(op_counter, tensor_counter, tensor_name, pre_key, pre_val))
                new_grid_sample_table[tensor_name] = pre_val
                for value_info in onnx_model.graph.value_info:
                    if value_info.name == tensor_name:
                        logger.info("[{}-{}] tensor_{}: has value_info {}".format(op_counter, tensor_counter, tensor_name, value_info))
                tensor_counter += 1
            op_counter += 1
    print_table(new_grid_sample_table)


if __name__ == "__main__":
    plogging.init("./", "grid_sample_value_info")
    logger = plogging.get_logger()
    # print_table()

    onnx_model_path = "../../onnx/transformer_op13_folded.onnx"
    onnx_model = onnx.load(onnx_model_path)
    refresh_table(onnx_model)


