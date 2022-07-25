from collections import OrderedDict
import torch, sys
import torch.nn as nn
from panoptic_bev.utils.sequence import pad_packed_images
from panoptic_bev.utils.parallel.packed_sequence import PackedSequence
import torch.utils.checkpoint as checkpoint
from panoptic_bev.algos.po_fusion_ts import po_inference
from panoptic_bev.algos.po_fusion_onnx import Pofusion_ONNX
sys.path.append("/home/hugoliu/github/PanopticBEV/onnx/script")
from onnx_wrapper import OnnxWrapper
from panoptic_bev.utils import plogging
logger = plogging.get_logger()


NETWORK_INPUTS = ["img", "calib", "extrinsics", "valid_msk"]
#False: turn_off jit, True: use jit inference
g_toggle_body_jit = False
g_toggle_transformer_jit = False
g_toggle_rpn_jit = False
g_toggle_roi_jit = False
g_toggle_semantic_jit = False
g_toggle_po_jit = False

g_toggle_rpn_onnx = False
g_toggle_roi_onnx = False
g_toggle_po_onnx = False

class PanopticBevNetTs(nn.Module):
    def __init__(self,
                 body,
                 transformer,
                 rpn_head,
                 roi_head,
                 sem_head,
                 transformer_algo,
                 rpn_algo,
                 inst_algo,
                 sem_algo,
                 po_fusion_algo,
                 dataset,
                 classes=None,
                 front_vertical_classes=None,  # In the frontal view
                 front_flat_classes=None,  # In the frontal view
                 bev_vertical_classes=None,  # In the BEV
                 bev_flat_classes=None,  # In the BEV
                 out_shape=None,
                 tfm_scales=None):
        super(PanopticBevNetTs, self).__init__()
        
        self.body_jit_path = "../jit/body_encoder.pt"
        self.body = body
        # Backbone
        if g_toggle_body_jit:
            self.body_jit = torch.jit.load(self.body_jit_path)
            logger.debug("load encoder: {}".format(self.body_jit_path))
        self.body_onnx_path = "../onnx/body_encoder_op13.onnx"

        # Transformer
        self.transformer = transformer
        self.transformer_jit_path = "../jit/ms_transformer.pt"
        if g_toggle_transformer_jit:
            self.transformer_jit = torch.jit.load(self.transformer_jit_path)
            logger.debug("load transformer: {}".format(self.transformer_jit_path))
        self.transformer_onnx_path = "../onnx/transformer_op13.onnx"

        # Modules
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head

        # Algorithms
        self.transformer_algo = transformer_algo
        self.rpn_algo = rpn_algo
        self.inst_algo = inst_algo
        self.sem_algo = sem_algo
        self.po_fusion_algo = po_fusion_algo

        self.rpn_algo_jit_path = "../jit/rpn_algo.pt"
        if g_toggle_rpn_jit:
            self.rpn_algo_jit = torch.jit.load(self.rpn_algo_jit_path)
            logger.debug("load rpn_algo: {}".format(self.rpn_algo_jit_path))
        self.rpn_algo_onnx_path = "../onnx/rpn_algo_op13.onnx"
        if g_toggle_rpn_onnx:
            self.rpn_algo_onnx = OnnxWrapper()
            self.rpn_algo_onnx.load_onnx_model(self.rpn_algo_onnx_path)

        self.roi_algo_jit_path = "../jit/roi_algo.pt"
        if g_toggle_roi_jit:
            self.roi_algo_jit = torch.jit.load(self.roi_algo_jit_path)
            logger.debug("load roi_algo: {}".format(self.roi_algo_jit_path))
        self.roi_algo_onnx_path = "../onnx/roi_algo_op13.onnx"
        if g_toggle_roi_onnx:
            self.roi_algo_onnx = OnnxWrapper()
            self.roi_algo_onnx.load_onnx_model(self.roi_algo_onnx_path)

        self.sem_algo_jit_path = "../jit/sem_algo.pt"
        if g_toggle_semantic_jit:
            self.sem_algo_jit = torch.jit.load(self.sem_algo_jit_path)
            logger.debug("load sem_algo: {}".format(self.sem_algo_jit_path))

        self.sem_algo_onnx_path = "../onnx/sem_algo_op13.onnx"
        self.po_fusion_jit_path = "../jit/po_fusion.pt"
        if g_toggle_po_jit:
            self.po_fusion_jit = torch.jit.load(self.po_fusion_jit_path)
            logger.debug("load po_fusion: {}".format(self.po_fusion_jit_path))
       
        self.po_fusion_onnx_path = "../onnx/po_fusion_op13.onnx"
        if g_toggle_po_onnx:
            self.po_fusion_onnx = OnnxWrapper()
            self.po_fusion_onnx.load_onnx_model(self.po_fusion_onnx_path)

        # Params
        self.dataset = dataset
        self.num_stuff = classes["stuff"]
        self.front_vertical_classes = front_vertical_classes
        self.front_flat_classes = front_flat_classes
        self.bev_vertical_classes = bev_vertical_classes
        self.bev_flat_classes = bev_flat_classes
        self.out_shape=out_shape
        self.tfm_scales=tfm_scales
        self.ms_bev_0 = []
        W, Z = out_shape
        Z *= 2 # for multi_view, double Z_out
        self.img_size = torch.Size([W, Z])
        self.img_size_t = torch.tensor([W, Z])
        self.valid_size = [torch.tensor([W, Z])]
        self.valid_size_t = torch.stack(self.valid_size)
        for idx, scale in enumerate(tfm_scales):
            ms_bev_tmp = torch.zeros(1, 256, int(W/scale), int(Z/scale))
            logger.debug("{}- scale: {} append ms_bev: {}".format(idx, scale, ms_bev_tmp.shape))
            self.ms_bev_0.append(ms_bev_tmp)

        self.po_fusion = Pofusion_ONNX(self.img_size_t)
        # self.po_fusion_script = torch.jit.script(self.po_fusion)
        # torch.jit.save(self.po_fusion_script, self.po_fusion_jit_path)

    def load_trained_params(self):
        self.rpn_algo.set_head(self.rpn_head)

    def forward(self, img):
        
        # logger.info("img PackedSequence __len__(): {}, extrinsics len: {}, valid_msk len: {}".format(img.__len__(), len(extrinsics), len(valid_msk)))
            # logger.debug("panoptic_bev input, img: {},  cat:{}, iscrowd: {}, bbx: {}, do_loss: {}, do_prediction: {}".format(img, bev_msk, front_msk, weights_msk, cat, iscrowd, bbx, do_loss, do_prediction))

        # valid_size = [torch.Size([896, 768*2])] # for multi_view, double Z_out
        # img_size = torch.Size([896, 768*2]) # for multi_view, double Z_out

        # Get some parameters
        # pad img list into torch.Size([N, 3, 448, 768]), originally N=1
        ms_bev = []
        for idx, f in enumerate(self.ms_bev_0):
            ms_bev.append(self.ms_bev_0[idx].to(img.device))

        # just keep training process unchanged
        # for idx, (image, intrin, extrin, msk) in enumerate(zip(img[0], calib[0], extrinsics[0], valid_msk[0])):
        b, n, c, h, w = img.shape
        for idx in range(n):
            image = img[:, idx] #[1, 3, 448, 768]
            # intrin = calib[:, idx] #[1, 3, 3]
            # extrin = extrinsics[0, idx] #[2, 3]
            # msk = valid_msk[0, idx] #[896, 768]
            logger.debug("process camera-{}, img: {}".format(idx, image.shape))
            # Get the image features
            if g_toggle_body_jit:
                ms_feat = self.body_jit(image)
            else:
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
            if g_toggle_transformer_jit:
                ms_bev_tmp = self.transformer_jit(ms_feat, idx)
            else:
                ms_bev_tmp = self.transformer(ms_feat, idx)
            # transformer_ts = torch.jit.trace(self.transformer, (ms_feat, intrin, extrin, msk), check_trace=True)
            # torch.jit.save(transformer_ts, self.transformer_jit_path)
            # sys.exit(0)

            # torch.onnx.export(self.transformer, (ms_feat, idx, extrin, msk), self.transformer_onnx_path, opset_version=13, verbose=True, custom_opsets={"custom_domain": 1}, do_constant_folding=True)
            # sys.exit(0)


            # if ms_bev == None:
            #     ms_bev = ms_bev_tmp
            # else:
            for i, f in enumerate(ms_bev_tmp):
                ms_bev[i] += f
                # logger.debug("ms_bev[{}] shape: {}".format(i, ms_bev[i].shape))

            del ms_feat, ms_bev_tmp

        # RPN Part
        if g_toggle_rpn_jit:
            proposals = self.rpn_algo_jit(ms_bev)
        elif g_toggle_rpn_onnx:
            proposals = self.rpn_algo_onnx.run(ms_bev)
        else:
            # self.rpn_algo.set_head(self.rpn_head)
            proposals = self.rpn_algo(ms_bev)
        # rpn_algo_ts = torch.jit.trace(self.rpn_algo, (ms_bev, self.valid_size), check_trace=True)
        # torch.jit.save(rpn_algo_ts, self.rpn_algo_jit_path)
        # sys.exit(0)

        # torch.onnx.export(self.rpn_algo, (ms_bev), self.rpn_algo_onnx_path, opset_version=13, verbose=True, do_constant_folding=True)
        # sys.exit(0)

        # ROI Part
        if g_toggle_roi_jit:
            bbx_pred, cls_pred, obj_pred, roi_msk_logits = self.roi_algo_jit(ms_bev[0], ms_bev[1], ms_bev[2], ms_bev[3], proposals)
        elif g_toggle_roi_onnx:
            bbx_pred, cls_pred, obj_pred, roi_msk_logits = self.roi_algo_onnx.run([ms_bev[0], ms_bev[1], ms_bev[2], ms_bev[3], proposals])
        else:
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
        if g_toggle_semantic_jit:
            sem_pred, sem_logits = self.sem_algo_jit(ms_bev)
        else:
            # sem_pred, sem_logits, _ = self.sem_algo.inference(self.sem_head, ms_bev, self.valid_size, self.img_size)
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

        if g_toggle_po_jit:
            po_pred = self.po_fusion_jit(sem_logits, roi_msk_logits, bbx_pred, cls_pred)
        elif g_toggle_po_onnx:
            # onnxruntime load model failed, problem with torch.unique, torch.loop
            po_pred_seamless, po_cls, po_iscrowd = self.po_fusion_onnx.run([sem_logits, roi_msk_logits, bbx_pred, cls_pred])
        else:
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

        # Prepare outputs
        # PREDICTIONS
        result = OrderedDict()
        result['bbx_pred'] = bbx_pred[0]
        result['cls_pred'] = cls_pred[0]
        result['obj_pred'] = obj_pred[0]
        result["sem_pred"] = sem_pred
        logger.info("panoptic_bev output, bbx_pred: {}, cls_pred: {}, obj_pred: {}, sem_pred: {}".format(result['bbx_pred'].shape, result['cls_pred'].shape, result['obj_pred'].shape, result["sem_pred"].shape))
        result['po_pred'] = po_pred[0]
        result['po_class'] = po_pred[1]
        result['po_iscrowd'] = po_pred[2]
        logger.info("panoptic_bev output, po_pred: {}, po_class: {}, po_iscrowd: {}".format(result['po_pred'].shape, result['po_class'], result['po_iscrowd']))

        return result
