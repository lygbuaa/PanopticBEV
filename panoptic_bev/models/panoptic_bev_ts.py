from collections import OrderedDict
import torch
import torch.nn as nn
from panoptic_bev.utils.sequence import pad_packed_images
from panoptic_bev.utils.parallel.packed_sequence import PackedSequence
import torch.utils.checkpoint as checkpoint
from panoptic_bev.utils import plogging
logger = plogging.get_logger()


NETWORK_INPUTS = ["img", "calib", "extrinsics", "valid_msk"]

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

        # Backbone
        self.body = body

        # Transformer
        self.transformer = transformer

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
        self.valid_size = [torch.Size([W, Z])]
        for idx, scale in enumerate(tfm_scales):
            ms_bev_tmp = torch.zeros(1, 256, int(W/scale), int(Z/scale))
            logger.debug("{}- scale: {} append ms_bev: {}".format(idx, scale, ms_bev_tmp.shape))
            self.ms_bev_0.append(ms_bev_tmp)

    def forward(self, img, calib=None, extrinsics=None, valid_msk=None):
        result = OrderedDict()
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
            intrin = calib[:, idx] #[1, 3, 3]
            extrin = extrinsics[0, idx] #[2, 3]
            msk = valid_msk[0, idx] #[896, 768]
            logger.debug("process camera-{}, img: {}, intrinsics: {}, extrinsics: {}, msk: {}".format(idx, image.shape, intrin.shape, extrin, msk.shape))
            # Get the image features
            ms_feat = self.body(image)

            # Transform from the front view to the BEV and upsample the height dimension
            ms_bev_tmp, _, _, _ = self.transformer(ms_feat, intrin, extrin, msk)
            # if ms_bev == None:
            #     ms_bev = ms_bev_tmp
            # else:
            for i, f in enumerate(ms_bev_tmp):
                ms_bev[i] += f
                # logger.debug("ms_bev[{}] shape: {}".format(i, ms_bev[i].shape))

            del ms_feat, ms_bev_tmp

        # RPN Part
        proposals = self.rpn_algo.inference(self.rpn_head, ms_bev, self.valid_size, self.training)

        # ROI Part
        bbx_pred, cls_pred, obj_pred, msk_pred, roi_msk_logits = self.inst_algo.inference(self.roi_head, ms_bev,
                                                                                              proposals, self.valid_size,
                                                                                              self.img_size)

        # Segmentation Part
        sem_pred, sem_logits, sem_feat = self.sem_algo.inference(self.sem_head, ms_bev, self.valid_size, self.img_size)

        # Panoptic Fusion. Fuse the semantic and instance predictions to generate a coherent output
        # The first channel of po_pred contains the semantic labels
        # The second channel contains the instance masks with the instance label being the corresponding semantic label
        po_pred, _, _ = self.po_fusion_algo.inference(sem_logits, roi_msk_logits, bbx_pred, cls_pred, self.img_size)
        logger.debug("po_fusion input: sem_logits: {}, roi_msk_logits: {}, bbx_pred: {}, cls_pred: {}".format(sem_logits[0].shape, roi_msk_logits[0].shape, bbx_pred[0].shape, cls_pred[0].shape))

        # Prepare outputs
        # PREDICTIONS
        result['bbx_pred'] = bbx_pred.contiguous[0]
        result['cls_pred'] = cls_pred.contiguous[0]
        result['obj_pred'] = obj_pred.contiguous[0]
        result['msk_pred'] = msk_pred.contiguous[0]
        result["sem_pred"] = sem_pred.contiguous[0]
        # result['sem_logits'] = sem_logits
        # result['vf_logits'] = vf_logits_list
        # result['v_region_logits'] = v_region_logits_list
        # result['f_region_logits'] = f_region_logits_list
        logger.info("panoptic_bev output, bbx_pred: {}, cls_pred: {}, obj_pred: {}, msk_pred: {}, roi_msk_logits: {}".format(result['bbx_pred'].shape, result['cls_pred'].shape, result['obj_pred'].shape, result['msk_pred'].shape, roi_msk_logits[0].shape))
        result['po_pred'] = po_pred[0].contiguous[0]
        result['po_class'] = po_pred[1].contiguous[0]
        result['po_iscrowd'] = po_pred[2].contiguous[0]
        logger.info("panoptic_bev output, po_pred: {}, po_class: {}, po_iscrowd: {}".format(result['po_pred'].shape, result['po_class'], result['po_iscrowd']))

        return result
