from collections import OrderedDict
import torch, sys
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List
from panoptic_bev.algos.po_fusion_onnx import Pofusion_ONNX
from panoptic_bev.algos.po_fusion_ts import po_inference
from trt.scripts.onnx2trt_test import TrtWrapper
from panoptic_bev.utils import plogging
logger = plogging.get_logger()

'''
NETWORK_INPUTS = ["img"]
#False: turn_off jit, True: use jit inference
g_toggle_body_jit = False
g_toggle_transformer_jit = False
g_toggle_rpn_jit = False
g_toggle_roi_jit = False
g_toggle_semantic_jit = False
g_toggle_po_jit = False

class BackboneTrt(nn.Module):
    def __init__(self, body, transformer, out_shape, tfm_scales):
        super(BackboneTrt, self).__init__()
        # encoder
        self.body = body
        # Transformer
        self.transformer = transformer
        W, Z = out_shape
        Z *= 2 # for multi_view, double Z_out
        self.ms_bev_0 = []
        for idx, scale in enumerate(tfm_scales):
            ms_bev_tmp = torch.zeros(1, 256, int(W/scale), int(Z/scale))
            logger.debug("{}- scale: {} append ms_bev: {}".format(idx, scale, ms_bev_tmp.shape))
            self.ms_bev_0.append(ms_bev_tmp)
        # set N=6, avoid onnx-tensorrt error: b, n, c, h, w = img.shape
        self.N = 6
        # self.transformer_script = torch.jit.script(self.transformer)
        # torch.jit.save(self.transformer_script, "../jit/transformer_script.pt")

        self.encoder_trt_path = "../trt/encoder_fp16.trt"
        self.encoder_trt_wrapper = TrtWrapper(self.encoder_trt_path)
        self.encoder_output_shape_list = [[1, 160, 112, 192], [1, 160, 56, 96], [1, 160, 28, 48], [1, 160, 14, 24], [1, 160, 7, 12]]
        self.transformer_trt_path = "../trt/transformer_lite_fp16.trt"
        self.transformer_trt_wrapper = TrtWrapper(self.transformer_trt_path)
        #[[1, 256, 224, 384], [1, 256, 112, 192], [1, 256, 56, 96], [1, 256, 28, 48]]
        self.transformer_output_shape_list = [[1, 256, 224, 192], [1, 256, 112, 96], [1, 256, 56, 48], [1, 256, 28, 24]]
        self.transformer_onnx_path = "../onnx/transformer_op13_lite.onnx"
    # ms_bev:List, 
    def forward(self, img:torch.Tensor, idx:int=0):
        # just keep training process unchanged
        # b, n, c, h, w = img.shape
        # for idx in range(N):
            # image = img[:, idx] #[1, 3, 448, 768]
            # logger.debug("process camera-{}, img: {}".format(idx, image.shape))
            # Get the image features
        # ms_feat = self.body(img)
        ms_feat = self.encoder_trt_wrapper.run([img], self.encoder_output_shape_list)
        # torch.onnx.export(self.body, (image), "../onnx/body_encoder_op13.onnx", opset_version=13, verbose=True, do_constant_folding=False)
        # sys.exit(0)

        # ms_bev_tmp = self.transformer(ms_feat, idx)
        ms_bev_tmp = self.transformer_trt_wrapper.run([[ms_feat[0], ms_feat[1], ms_feat[2], ms_feat[3]], idx], self.transformer_output_shape_list)
        # torch.onnx.export(self.transformer, (ms_feat, idx), self.transformer_onnx_path, opset_version=13, verbose=True, custom_opsets={"custom_domain": 1}, do_constant_folding=True)
        # sys.exit(0)

        # ms_bev_final = self.transformer.run_neck(ms_bev_tmp, idx)

        # for i, f in enumerate(ms_bev_tmp):
        #     ms_bev[i] += f

        return ms_bev_tmp

class TransformerNeck(nn.Module):
    def __init__(self, bev_params=None, Z_out=None, W_out=None, tfm_scales=None, initializer_generator=None):
        super(TransformerNeck, self).__init__()
        self.ms_valid_mask_list = []
        self.ms_affine_grid_list = []
        self.ms_z_out_list = []
        for scale_idx, scale in enumerate(tfm_scales):
            img_scale = 1/scale
            self.Z_out = int(Z_out * img_scale)
            self.W_out = int(W_out * img_scale)
            self.BEV_RESOLUTION  = bev_params['cam_z'] / bev_params['f'] / img_scale

            valid_mask_list = initializer_generator.generate_valid_mask(z_out=self.Z_out, w_out=self.W_out)
            affine_grid_list = initializer_generator.generate_affine_grid(z_out=self.Z_out, w_out=self.W_out, bev_resolution=self.BEV_RESOLUTION)
            self.ms_valid_mask_list.append(valid_mask_list)
            self.ms_affine_grid_list.append(affine_grid_list)
            self.ms_z_out_list.append(self.Z_out)
            logger.info("TransformerNeck append valid_mask_list & affine_grid_list for scale: {}-{}".format(scale_idx, scale))

    def run_neck(self, feat_merged:torch.Tensor, scale_idx:int, img_idx:int) -> torch.Tensor:
        msk_t = self.ms_valid_mask_list[scale_idx][img_idx].to(feat_merged.device)
        feat_merged = torch.mul(feat_merged, msk_t)
        # double bev size, padding on last dim, left_side
        feat_merged = F.pad(feat_merged, (self.ms_z_out_list[scale_idx], 0), mode="constant", value=0.0)

        grid = self.ms_affine_grid_list[scale_idx][img_idx].to(feat_merged.device)
        feat_merged = F.grid_sample(feat_merged, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return feat_merged

    def forward(self, ms_bev:List[torch.Tensor], img_idx:int) -> List[torch.Tensor]:
        ms_feat_final = []
        for scale_idx, in_feat in enumerate(ms_bev):
            out_feat = self.run_neck(in_feat, scale_idx, img_idx)
            ms_feat_final.append(out_feat)
            # logger.info("[{}-{}] in_feat: {}, out_feat: {}".format(scale_idx, img_idx, in_feat.shape, out_feat.shape))
        return ms_feat_final

class HeadsJit(nn.Module):
    def __init__(self, rpn_head, roi_head, sem_head, rpn_algo, inst_algo, sem_algo, out_shape):
        super(HeadsJit, self).__init__()
        self.out_shape=out_shape
        W, Z = out_shape
        Z *= 2 # for multi_view, double Z_out
        self.img_size = torch.Size([W, Z])
        self.img_size_t = torch.tensor([W, Z])
        self.po_fusion = Pofusion_ONNX(self.img_size_t)
        # Modules
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head
        # Algorithms
        self.rpn_algo = rpn_algo
        self.inst_algo = inst_algo
        self.sem_algo = sem_algo

        # self.roi_algo_jit = torch.jit.script(self.inst_algo)
        # self.po_fusion_jit = torch.jit.script(self.po_fusion)

    def forward(self, ms_bev):
        # RPN Part
        proposals = self.rpn_algo(ms_bev)
        # ROI Part
        bbx_pred, cls_pred, obj_pred, roi_msk_logits = self.inst_algo(ms_bev[0], ms_bev[1], ms_bev[2], ms_bev[3], proposals)
        # bbx_pred, cls_pred, obj_pred, roi_msk_logits = self.roi_algo_jit(ms_bev[0], ms_bev[1], ms_bev[2], ms_bev[3], proposals)
        # Segmentation Part
        sem_pred, sem_logits = self.sem_algo(ms_bev)
        logger.debug("po_fusion input: sem_logits: {}-{}, roi_msk_logits: {}-{}, bbx_pred: {}-{}, cls_pred: {}-{}".format(sem_logits.shape, sem_logits.dtype, roi_msk_logits.shape, roi_msk_logits.dtype, bbx_pred.shape, bbx_pred.dtype, cls_pred.shape, cls_pred.dtype))
        # po_pred_seamless, po_cls, po_iscrowd = self.po_fusion(sem_logits, roi_msk_logits, bbx_pred, cls_pred)
        # po_pred_seamless, po_cls, po_iscrowd = self.po_fusion_jit(sem_logits, roi_msk_logits, bbx_pred, cls_pred)
        po_pred_seamless, po_cls, po_iscrowd = po_inference(sem_logits, roi_msk_logits, bbx_pred, cls_pred, self.img_size_t)
        return [bbx_pred[0], cls_pred[0], obj_pred[0], sem_pred, po_pred_seamless, po_cls, po_iscrowd]
'''

class PanopticBevNetTrt(nn.Module):
    def __init__(self, out_shape, tfm_scales):
        super(PanopticBevNetTrt, self).__init__()
        W, Z = out_shape
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
    '''
    def __init__(self,
                 body,
                 transformer,
                 rpn_head,
                 roi_head,
                 sem_head,
                 rpn_algo,
                 inst_algo,
                 sem_algo,
                 po_fusion_algo,
                 out_shape,
                 tfm_scales,
                 bev_params,
                 initializer_generator):
        super(PanopticBevNetTrt, self).__init__()
        # encoder
        self.body = body
        # Transformer
        self.transformer = transformer
        W, Z = out_shape

        # Modules
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.inst_algo = inst_algo
        self.sem_algo = sem_algo
        self.po_fusion_algo = po_fusion_algo

        # Params
        self.out_shape=out_shape
        self.tfm_scales=tfm_scales
        Z *= 2 # for multi_view, double Z_out
        self.img_size = torch.Size([W, Z])
        self.img_size_t = torch.tensor([W, Z])
        self.ms_bev_0 = []
        for idx, scale in enumerate(tfm_scales):
            ms_bev_tmp = torch.zeros(1, 256, int(W/scale), int(Z/scale))
            logger.debug("{}- scale: {} append ms_bev: {}".format(idx, scale, ms_bev_tmp.shape))
            self.ms_bev_0.append(ms_bev_tmp)

        self.po_fusion = Pofusion_ONNX(self.img_size_t)
        # self.po_fusion_script = torch.jit.script(self.po_fusion)
        # torch.jit.save(self.po_fusion_script, self.po_fusion_jit_path)

        self.backbone_trt = BackboneTrt(body, transformer, out_shape, tfm_scales)
        self.heads_jit = HeadsJit(rpn_head, roi_head, sem_head, rpn_algo, inst_algo, sem_algo, out_shape)

        self.backbone_onnx_path = "../onnx/backbone_lite_op13.onnx"
        self.heads_jit_path = "../jit/heads_jit.pt"
        self.transformer_neck_jit_path = "../jit/transformer_neck_jit.pt"

        self.backbone_trt_path = "../trt/backbone_lite_fp16.trt"
        self.trt_wrapper = TrtWrapper(self.backbone_trt_path)
        #[[1, 256, 224, 384], [1, 256, 112, 192], [1, 256, 56, 96], [1, 256, 28, 48]]
        self.backbone_output_shape_list = [[1, 256, 224, 192], [1, 256, 112, 96], [1, 256, 56, 48], [1, 256, 28, 24]]

        # self.transformer_neck = TransformerNeck(bev_params=bev_params, Z_out=Z, W_out=W, tfm_scales=tfm_scales, initializer_generator=initializer_generator)
        # self.transformer_neck_script = torch.jit.script(self.transformer_neck)
        # torch.jit.save(self.transformer_neck_script, self.transformer_neck_jit_path)
        self.transformer_neck_jit_model = torch.jit.load(self.transformer_neck_jit_path)
        self.heads_jit_model = torch.jit.load(self.heads_jit_path)
    '''
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
    '''
    def forward_0(self, img):
        ms_bev = []
        for idx, f in enumerate(self.ms_bev_0):
            ms_bev.append(self.ms_bev_0[idx].to(img.device))

        for idx in range(6):
            image = img[:, idx] #[1, 3, 448, 768]
            ms_bev_tmp = self.backbone_trt(image, idx)
            # ms_bev_tmp = self.trt_wrapper.run([image, idx], self.backbone_output_shape_list)
            # ms_bev_final = self.transformer.run_neck(ms_bev_tmp, idx)
            ms_bev_final = self.transformer_neck_jit_model(ms_bev_tmp, idx)

            # self.transformer_neck_script = torch.jit.trace(self.transformer.run_neck, (ms_bev_tmp, idx), check_trace=True)
            # torch.jit.save(self.transformer_neck_script, self.transformer_neck_jit_path)

            # torch.onnx.export(self.backbone_trt, (image, idx), self.backbone_onnx_path, opset_version=13, verbose=True, do_constant_folding=False)
            # sys.exit(0)
            for i, f in enumerate(ms_bev_final):
                ms_bev[i] += f

        # results = self.heads_jit(ms_bev)
        results = self.heads_jit_model(ms_bev)

        # heads_jit_model = torch.jit.trace(self.heads_jit, ([ms_bev]), check_trace=True)
        # torch.jit.save(heads_jit_model, self.heads_jit_path)
        # sys.exit(0)

        return results
    '''
    '''
    def forward_bak(self, img):
        # valid_size = [torch.Size([896, 768*2])] # for multi_view, double Z_out
        # img_size = torch.Size([896, 768*2]) # for multi_view, double Z_out

        # Get some parameters
        ms_bev = []
        for idx, f in enumerate(self.ms_bev_0):
            ms_bev.append(self.ms_bev_0[idx].to(img.device))

        # just keep training process unchanged
        b, n, c, h, w = img.shape
        for idx in range(n):
            image = img[:, idx] #[1, 3, 448, 768]
            # intrin = calib[:, idx] #[1, 3, 3]
            # extrin = extrinsics[0, idx] #[2, 3]
            # msk = valid_msk[0, idx] #[896, 768]
            logger.debug("process camera-{}, img: {}".format(idx, image.shape))
            # Get the image features
            ms_feat = self.body(image)

            # debug_str = "ms_feat: "
            # for i, f in enumerate(ms_feat):
            #     tmp_str = " {}-{},".format(i, f.shape)
            #     debug_str += tmp_str
            # logger.debug(debug_str)

            # body_ts = torch.jit.trace(self.body, (image), check_trace=True)
            # torch.jit.save(body_ts, self.body_jit_path)
            # sys.exit(0)

            # torch.onnx.export(self.body, image, self.body_onnx_path, opset_version=13, verbose=True, do_constant_folding=True)
            # sys.exit(0)

            # Transform from the front view to the BEV and upsample the height dimension
            ms_bev_tmp = self.transformer(ms_feat, idx)
            # transformer_ts = torch.jit.trace(self.transformer, (ms_feat, intrin, extrin, msk), check_trace=True)
            # torch.jit.save(transformer_ts, self.transformer_jit_path)
            # sys.exit(0)

            # torch.onnx.export(self.transformer, (ms_feat, idx), self.transformer_onnx_path, opset_version=13, verbose=True, custom_opsets={"custom_domain": 1}, do_constant_folding=True)
            # sys.exit(0)


            # if ms_bev == None:
            #     ms_bev = ms_bev_tmp
            # else:
            for i, f in enumerate(ms_bev_tmp):
                ms_bev[i] += f
                # logger.debug("ms_bev[{}] shape: {}".format(i, ms_bev[i].shape))

            del ms_feat, ms_bev_tmp

        # RPN Part
        proposals = self.rpn_algo(ms_bev)
        # rpn_algo_ts = torch.jit.trace(self.rpn_algo, (ms_bev, self.valid_size), check_trace=True)
        # torch.jit.save(rpn_algo_ts, self.rpn_algo_jit_path)
        # sys.exit(0)

        # torch.onnx.export(self.rpn_algo, (ms_bev), self.rpn_algo_onnx_path, opset_version=13, verbose=True, do_constant_folding=True)
        # sys.exit(0)

        # ROI Part
        bbx_pred, cls_pred, obj_pred, roi_msk_logits = self.inst_algo(ms_bev[0], ms_bev[1], ms_bev[2], ms_bev[3], proposals)
        # roi_algo_jit = torch.jit.script(self.inst_algo)
        # torch.jit.save(roi_algo_jit, self.roi_algo_jit_path)
        # torch.onnx.export(
        #    model=roi_algo_jit, 
        #    args=(ms_bev[0], ms_bev[1], ms_bev[2], ms_bev[3], proposals),
        #    f=self.roi_algo_onnx_path,
        #    input_names=["ms_bev_0", "ms_bev_1", "ms_bev_2", "ms_bev_3", "proposals"],
        #    output_names=["bbx_pred", "cls_pred", "obj_pred", "roi_msk_logits"],
        #    dynamic_axes={
        #            "ms_bev_0": [0],
        #            "ms_bev_1": [0],
        #            "ms_bev_2": [0],
        #            "ms_bev_3": [0],
        #            "proposals": [0, 1],
        #            "bbx_pred": [1],
        #            "cls_pred": [1],
        #            "obj_pred": [1], 
        #            "roi_msk_logits": [1],
        #        },
        #    opset_version=13, verbose=True, do_constant_folding=True)
        # sys.exit(0)

        # Segmentation Part
        sem_pred, sem_logits = self.sem_algo(ms_bev)
        # sem_algo_ts = torch.jit.trace(self.sem_algo, ([ms_bev]), check_trace=True)
        # torch.jit.save(sem_algo_ts, self.sem_algo_jit_path)
        # sys.exit(0)

        # torch.onnx.export(self.sem_algo, ms_bev, self.sem_algo_onnx_path, opset_version=13, verbose=True, do_constant_folding=True)
        # sys.exit(0)

        # Panoptic Fusion. Fuse the semantic and instance predictions to generate a coherent output
        # The first channel of po_pred contains the semantic labels
        # The second channel contains the instance masks with the instance label being the corresponding semantic label
        # po_pred, _, _ = self.po_fusion_algo.inference(sem_logits, roi_msk_logits, bbx_pred, cls_pred, self.img_size)

        # bbx_pred = torch.stack(bbx_pred, dim=0)
        # cls_pred = torch.stack(cls_pred, dim=0)
        # roi_msk_logits = torch.stack(roi_msk_logits, dim=0)
        logger.debug("po_fusion input: sem_logits: {}-{}, roi_msk_logits: {}-{}, bbx_pred: {}-{}, cls_pred: {}-{}".format(sem_logits.shape, sem_logits.dtype, roi_msk_logits.shape, roi_msk_logits.dtype, bbx_pred.shape, bbx_pred.dtype, cls_pred.shape, cls_pred.dtype))

        # sem_logits = torch.rand([1, 10, 896, 1536], dtype=torch.float)
        # roi_msk_logits = torch.rand([1, 11, 4, 28, 28], dtype=torch.float)
        # bbx_pred = torch.rand([1, 11, 4], dtype=torch.float)
        # cls_pred = torch.rand([1, 11], dtype=torch.float).to(torch.int64)

        # po_pred = po_inference(sem_logits, roi_msk_logits, bbx_pred, cls_pred, self.img_size_t)
        po_pred_seamless, po_cls, po_iscrowd = self.po_fusion(sem_logits, roi_msk_logits, bbx_pred, cls_pred)

        # po_fusion_jit = torch.jit.script(self.po_fusion)
        # torch.jit.save(po_fusion_jit, self.po_fusion_jit_path)
        # logger.debug("po_fusion_jit: {}".format(po_fusion_jit.graph))
        # torch.onnx.export(
        #     model=po_fusion_jit, 
        #     args=(sem_logits, roi_msk_logits, bbx_pred, cls_pred), 
        #     f=self.po_fusion_onnx_path, 
        #     input_names=["sem_logits", "roi_msk_logits", "bbx_pred", "cls_pred"],
        #     output_names=["po_pred_seamless", "po_cls", "po_iscrowd"],
        #     dynamic_axes={
        #             "roi_msk_logits": [1],
        #             "bbx_pred": [1],
        #             "cls_pred": [1],
        #           },
        #     opset_version=13, verbose=True, do_constant_folding=True)
        # sys.exit(0)

        return [bbx_pred[0], cls_pred[0], obj_pred[0], sem_pred, po_pred_seamless, po_cls, po_iscrowd]
        '''