from collections import OrderedDict
import torch, sys, time
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from panoptic_bev.algos.po_fusion_ts import po_inference
from panoptic_bev.utils import plogging
logger = plogging.get_logger()


NETWORK_INPUTS = ["img", "calib", "extrinsics", "valid_msk"]

class PanopticBevNetJIT(nn.Module):
    def __init__(self, out_shape=None, tfm_scales=None):
        super(PanopticBevNetJIT, self).__init__()
        
        self.body_jit_path = "../jit/body_encoder.pt"
        self.body_jit = torch.jit.load(self.body_jit_path)
        logger.debug("load encoder: {}".format(self.body_jit_path))

        # Transformer
        self.transformer_jit_path = "../jit/ms_transformer.pt"
        self.transformer_jit = torch.jit.load(self.transformer_jit_path)
        logger.debug("load transformer: {}".format(self.transformer_jit_path))

        self.rpn_algo_jit_path = "../jit/rpn_algo.pt"
        self.rpn_algo_jit = torch.jit.load(self.rpn_algo_jit_path)
        logger.debug("load rpn_algo: {}".format(self.rpn_algo_jit_path))

        self.roi_algo_jit_path = "../jit/roi_algo.pt"
        self.roi_algo_jit = torch.jit.load(self.roi_algo_jit_path)
        logger.debug("load roi_algo: {}".format(self.roi_algo_jit_path))

        self.sem_algo_jit_path = "../jit/sem_algo.pt"
        self.sem_algo_jit = torch.jit.load(self.sem_algo_jit_path)
        logger.debug("load sem_algo: {}".format(self.sem_algo_jit_path))

        self.po_fusion_jit_path = "../jit/po_fusion.pt"
        self.po_fusion_jit = torch.jit.load(self.po_fusion_jit_path)
        logger.debug("load po_fusion: {}".format(self.po_fusion_jit_path))

        # Params
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


    def forward(self, img:torch.Tensor, calib:torch.Tensor, extrinsics:torch.Tensor, valid_msk:torch.Tensor, LOOP:int=1):
        result = OrderedDict()
        # valid_size = [torch.Size([896, 768*2])] # for multi_view, double Z_out
        # img_size = torch.Size([896, 768*2]) # for multi_view, double Z_out

        # Get some parameters
        # pad img list into torch.Size([N, 3, 448, 768]), originally N=1
        ms_bev = []
        ms_feat = []
        for idx, f in enumerate(self.ms_bev_0):
            ms_bev.append(self.ms_bev_0[idx].to(img.device))

        # just keep training process unchanged
        # for idx, (image, intrin, extrin, msk) in enumerate(zip(img[0], calib[0], extrinsics[0], valid_msk[0])):
        b, n, c, h, w = img.shape
        for idx in range(n):
            image = img[:, idx] #[1, 3, 448, 768]
            intrin = calib[:, idx] #[1, 3, 3]
            extrin = extrinsics[0, idx] #[2, 3]
            msk = valid_msk[0, idx] #[896, 768]
            # print("process camera-{}, img: {}, intrinsics: {}, extrinsics: {}, msk: {}".format(idx, image.shape, intrin.shape, extrin, msk.shape))
            # Get the image features
            start_time = time.time()
            logger.debug("encoder-in, {}".format(start_time))
            for i in range(LOOP):
                ms_feat = self.body_jit(image)
            end_time = time.time()
            logger.debug("encoder-out, {}, average: {}".format(end_time, (end_time-start_time)/LOOP))

            # debug_str = "ms_feat: "
            # for i, f in enumerate(ms_feat):
            #     tmp_str = " {}-{},".format(i, f.shape)
            #     debug_str += tmp_str
            # logger.debug(debug_str)

            # Transform from the front view to the BEV and upsample the height dimension
            start_time = time.time()
            logger.debug("transformer-in, {}".format(start_time))
            for i in range(LOOP):
                ms_bev_tmp = self.transformer_jit(ms_feat, intrin, extrin, msk)
            end_time = time.time()
            logger.debug("transformer-out, {}, average: {}".format(end_time, (end_time-start_time)/LOOP))

            # if ms_bev == None:
            #     ms_bev = ms_bev_tmp
            # else:
            for i, f in enumerate(ms_bev_tmp):
                ms_bev[i] += f
                # logger.debug("ms_bev[{}] shape: {}".format(i, ms_bev[i].shape))

            del ms_feat, ms_bev_tmp

        # RPN Part
        start_time = time.time()
        logger.debug("rpn-in, {}".format(start_time))
        for i in range(LOOP):
            proposals = self.rpn_algo_jit(ms_bev, self.valid_size)
        end_time = time.time()
        logger.debug("rpn-out, {}, average: {}".format(end_time, (end_time-start_time)/LOOP))

        # ROI Part
        start_time = time.time()
        logger.debug("roi-in, {}".format(start_time))
        for i in range(LOOP):
            bbx_pred, cls_pred, obj_pred, roi_msk_logits = self.roi_algo_jit(ms_bev, proposals, self.valid_size, self.img_size_t)
        end_time = time.time()
        logger.debug("roi-out, {}, average: {}".format(end_time, (end_time-start_time)/LOOP))

        # Segmentation Part
        start_time = time.time()
        logger.debug("sem-in, {}".format(start_time))
        for i in range(LOOP):
            sem_pred, sem_logits = self.sem_algo_jit(ms_bev, self.valid_size_t, self.img_size_t)
        end_time = time.time()
        logger.debug("sem-out, {}, average: {}".format(end_time, (end_time-start_time)/LOOP))

        # Panoptic Fusion. Fuse the semantic and instance predictions to generate a coherent output
        # The first channel of po_pred contains the semantic labels
        # The second channel contains the instance masks with the instance label being the corresponding semantic label
        # po_pred, _, _ = self.po_fusion_algo.inference(sem_logits, roi_msk_logits, bbx_pred, cls_pred, self.img_size)
        bbx_pred = torch.stack(bbx_pred, dim=0)
        cls_pred = torch.stack(cls_pred, dim=0)
        roi_msk_logits = torch.stack(roi_msk_logits, dim=0)
        # print("po_fusion input: sem_logits: {}, roi_msk_logits: {}, bbx_pred: {}, cls_pred: {}".format(sem_logits.shape, roi_msk_logits.shape, bbx_pred, cls_pred))
        
        start_time = time.time()
        logger.debug("fusion-in, {}".format(start_time))
        for i in range(LOOP):
            po_pred = self.po_fusion_jit(sem_logits, roi_msk_logits, bbx_pred, cls_pred, self.img_size_t)
        end_time = time.time()
        logger.debug("fusion-out, {}, average: {}".format(end_time, (end_time-start_time)/LOOP))

        # Prepare outputs
        # PREDICTIONS
        result['bbx_pred'] = bbx_pred[0]
        result['cls_pred'] = cls_pred[0]
        result['obj_pred'] = obj_pred[0]
        result["sem_pred"] = sem_pred
        # logger.info("panoptic_bev output, bbx_pred: {}, cls_pred: {}, obj_pred: {}, sem_pred: {}".format(result['bbx_pred'].shape, result['cls_pred'].shape, result['obj_pred'].shape, result["sem_pred"].shape))
        result['po_pred'] = po_pred[0]
        result['po_class'] = po_pred[1]
        result['po_iscrowd'] = po_pred[2]
        # logger.info("panoptic_bev output, po_pred: {}, po_class: {}, po_iscrowd: {}".format(result['po_pred'].shape, result['po_class'], result['po_iscrowd']))

        return result