import torch
import torch.nn as nn
from trt.scripts.onnx2trt_test import TrtWrapper
from panoptic_bev.utils import plogging
logger = plogging.get_logger()

class PanopticBevModel(nn.Module):
    def __init__(self, out_shape, tfm_scales):
        super(PanopticBevModel, self).__init__()
        W, Z = out_shape

        # Params
        self.tfm_scales=tfm_scales
        Z *= 2 # for multi_view, double Z_out
        self.ms_bev_0 = []
        for idx, scale in enumerate(tfm_scales):
            ms_bev_tmp = torch.zeros(1, 256, int(W/scale), int(Z/scale))
            logger.debug("{}- scale: {} append ms_bev: {}".format(idx, scale, ms_bev_tmp.shape))
            self.ms_bev_0.append(ms_bev_tmp)

        self.encoder_trt_path = "../trt/encoder_fp16.trt"
        self.encoder_trt_wrapper = TrtWrapper(self.encoder_trt_path)
        self.encoder_output_shape_list = [[1, 160, 112, 192], [1, 160, 56, 96], [1, 160, 28, 48], [1, 160, 14, 24], [1, 160, 7, 12]]
        self.transformer_trt_path = "../trt/transformer_lite_fp16.trt"
        self.transformer_trt_wrapper = TrtWrapper(self.transformer_trt_path)
        self.transformer_output_shape_list = [[1, 256, 224, 192], [1, 256, 112, 96], [1, 256, 56, 48], [1, 256, 28, 24]]
        self.transformer_neck_jit_path = "../jit/transformer_neck_jit.pt"
        #load custom op: roi_sampling
        torch.ops.load_library("/home/hugoliu/github/PanopticBEV/panoptic_bev/utils/roi_sampling/_backend.cpython-38-x86_64-linux-gnu.so")
        self.heads_jit_path = "../jit/heads_jit.pt"
        self.transformer_neck_jit_model = torch.jit.load(self.transformer_neck_jit_path)
        self.heads_jit_model = torch.jit.load(self.heads_jit_path)

    def forward(self, img):
        ms_bev = []
        for idx, f in enumerate(self.ms_bev_0):
            ms_bev.append(self.ms_bev_0[idx].to(img.device))

        for idx in range(6):
            image = img[:, idx] #[1, 3, 448, 768]
            ms_feat = self.encoder_trt_wrapper.run([image], self.encoder_output_shape_list)
            ms_bev_tmp = self.transformer_trt_wrapper.run([[ms_feat[0], ms_feat[1], ms_feat[2], ms_feat[3]], idx], self.transformer_output_shape_list)
            ms_bev_final = self.transformer_neck_jit_model(ms_bev_tmp, idx)

            for i, f in enumerate(ms_bev_final):
                ms_bev[i] += f

        results = self.heads_jit_model(ms_bev)
        return results

if __name__ == "__main__":
    pass
