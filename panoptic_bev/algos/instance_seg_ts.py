import torch, sys
import torch.nn.functional as functional
from typing import Tuple, List
from math import log
from panoptic_bev.utils.bbx import shift_boxes
from panoptic_bev.utils.nms import nms
from panoptic_bev.utils.misc import Empty
from panoptic_bev.utils.parallel import PackedSequence
from panoptic_bev.utils.roi_sampling import roi_sampling
from panoptic_bev.utils import plogging
logger = plogging.get_logger()


@torch.jit.script
def g_shift_boxes(bbx:torch.Tensor, shift:torch.Tensor, dim:int=-1, scale_clip:float=log(1000. / 16.)):
    """Shift bounding boxes using the faster r-CNN formulas

    Each 4-vector of `bbx` and `shift` contain, respectively, bounding box coordiantes in "corners" form and shifts
    in the form `(dy, dx, dh, dw)`. The output is calculated according to the Faster r-CNN formulas:

        y_out = y_in + h_in * dy
        x_out = x_in + w_in * dx
        h_out = h_in * exp(dh)
        w_out = w_in * exp(dw)

    Parameters
    ----------
    bbx : torch.Tensor
        A tensor of bounding boxes with shape N_0 x ... x N_i = 4 x ... x N_n
    shift : torch.Tensor
        A tensor of shifts with shape N_0 x ... x N_i = 4 x ... x N_n
    dim : int
        The dimension i of the input tensors which contains the bounding box coordinates and the shifts
    scale_clip : float
        Maximum scale shift value to avoid exp overflow

    Returns
    -------
    bbx_out : torch.Tensor
        A tensor of shifted bounding boxes with shape N_0 x ... x N_i = 4 x ... x N_n

    """
    # yx_in, hw_in = corners_to_center_scale(*bbx.split(2, dim=dim))
    p0, p1 = bbx.split(2, dim=dim)
    yx_in = 0.5 * (p0 + p1)
    hw_in = p1 - p0
    # logger.debug("shift_boxes, bbx: {}, yx_in: {}, hw_in: {}".format(bbx.shape, yx_in.shape, hw_in.shape))
    dyx, dhw = shift.split(2, dim=dim)

    yx_out = yx_in + hw_in * dyx
    hw_out = hw_in * dhw.clamp(max=scale_clip).exp()

    hw_half = 0.5 * hw_out
    p0 = yx_out - hw_half
    p1 = yx_out + hw_half
    return torch.cat([p0, p1], dim=dim)

@torch.jit.script
def g_rois(
    x:torch.Tensor, proposals:torch.Tensor, proposals_idx:torch.Tensor, img_size:torch.Tensor, roi_size:torch.Tensor
    ) -> torch.Tensor:
    # stride = proposals.new([fs / os for fs, os in zip(x.shape[-2:], img_size)])
    stride = torch.stack([fs / os for fs, os in zip(x.shape[-2:], img_size)], dim=0).to(x.device)
    proposals = (proposals - 0.5) * stride.repeat(2) + 0.5
    # return roi_sampling(x, proposals, proposals_idx, roi_size)
    rois, msk = torch.ops.po_cpp_ops.po_roi(x, proposals, proposals_idx, roi_size)
    return rois

@torch.jit.script
def g_packed_sequence_contiguous(input_tensors:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    packed_tensors = []
    packed_idx = []
    for i, tensor in enumerate(input_tensors):
        if tensor is not None:
            packed_tensors.append(tensor)
            idx = tensor.new_full((tensor.size(0),), i, dtype=torch.long)
            packed_idx.append(idx)
            # logger.debug("packed_sequence: tensor: {}, idx: {}".format(tensor.shape, idx.shape))

    return torch.cat(packed_tensors, dim=0), torch.cat(packed_idx, dim=0)

@torch.jit.script
def g_prediction_generator(
    boxes:torch.Tensor, scores:torch.Tensor, nms_threshold:float=0.3, score_threshold:float=0.1, max_predictions:int=100
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    bbx_pred = []
    cls_pred = []
    obj_pred = []
    for bbx_i, obj_i in zip(boxes, scores):
        if bbx_i is None or obj_i is None:
            break

        # Class independent NMS
        bbx_all = bbx_i.reshape(-1, 4)
        scores_all = obj_i[:, 1:].reshape(-1)
        cls_all = torch.zeros(bbx_i.shape[0], bbx_i.shape[1], dtype=torch.long, device=bbx_i.device)
        for cls_id in range(bbx_i.shape[1]):
            cls_all[:, cls_id] = cls_id
        cls_all = cls_all.reshape(-1)

        # Filter out the low-scroing predictions
        idx = scores_all > score_threshold
        scores_all = scores_all[idx]
        bbx_all = bbx_all[idx]
        cls_all = cls_all[idx]

        # Filter empty predictions
        idx = (bbx_all[:, 2] > bbx_all[:, 0]) & (bbx_all[:, 3] > bbx_all[:, 1])
        if not idx.any().item():
            continue
        bbx_all = bbx_all[idx]
        scores_all = scores_all[idx]
        cls_all = cls_all[idx]

        # Do NMS
        # idx = nms(bbx_all.contiguous(), scores_all.contiguous(), threshold=self.nms_threshold, n_max=-1)
        idx = torch.ops.po_cpp_ops.po_nms(bbx_all.contiguous(), scores_all.contiguous(), nms_threshold, -1)
        if idx.numel() == 0:
            continue
        bbx_all = bbx_all[idx]
        scores_all = scores_all[idx]
        cls_all = cls_all[idx]

        if bbx_all.shape[0] == 0:
            break

        bbx_pred_i = bbx_all
        obj_pred_i = scores_all
        cls_pred_i = cls_all

        # Do post-NMS selection (if needed)
        if bbx_pred_i.size(0) > max_predictions:
            _, idx = obj_pred_i.topk(max_predictions)
            bbx_pred_i = bbx_pred_i[idx]
            cls_pred_i = cls_pred_i[idx]
            obj_pred_i = obj_pred_i[idx]

        # Save results
        bbx_pred.append(bbx_pred_i)
        cls_pred.append(cls_pred_i)
        obj_pred.append(obj_pred_i)

    return bbx_pred, cls_pred, obj_pred

class InstanceSegAlgoFPN_JIT(torch.nn.Module):
    """Instance segmentation algorithm for FPN networks

    Parameters
    ----------
    bbx_prediction_generator : faster_rcnn.PredictionGenerator
    msk_prediction_generator : mask_rcnn.PredictionGenerator
    proposal_matcher : faster_rcnn.ProposalMatcher
    bbx_loss : FasterRCNNLoss
    msk_loss : MaskRCNNLoss
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    bbx_reg_weights : sequence of float
        Weights assigned to the bbx regression coordinates
    canonical_scale : int
        Reference scale for ROI to FPN level assignment
    canonical_level : int
        Reference level for ROI to FPN level assignment
    roi_size : tuple of int
        Spatial size of the ROI features as `(height, width)`
    min_level : int
        First FPN level to work on
    levels : int
        Number of FPN levels to work on
    lbl_roi_size : tuple of int
        Spatial size of the ROI mask labels as `(height, width)`
    void_is_background : bool
        If True treat void areas as background in the instance mask loss instead of void
    """

    def __init__(self,
                #  head,
                 bbx_loss,
                 msk_loss,
                 bbx_reg_weights,
                 canonical_scale,
                 canonical_level,
                 roi_size,
                 min_level,
                 levels,
                 nms_threshold,
                 score_threshold,
                 max_predictions):
        super(InstanceSegAlgoFPN_JIT, self).__init__()
        # self.head = head
        self.bbx_loss = bbx_loss
        self.msk_loss = msk_loss
        self.bbx_reg_weights = bbx_reg_weights
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        # self.roi_size = roi_size
        self.roi_size = torch.tensor(roi_size, dtype=torch.int)
        self.min_level = min_level
        self.levels = levels
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_predictions = max_predictions
        self.head_bbx = torch.jit.load("../jit/roi_head_bbx.pt")
        self.head_msk = torch.jit.load("../jit/roi_head_msk.pt")
        # self.use_jit_head = True

    @staticmethod
    def _split_and_clip(boxes:torch.Tensor, scores:torch.Tensor, index:torch.Tensor, valid_size:List[torch.Tensor]):
        boxes_out, scores_out = [], []
        for img_id, valid_size_i in enumerate(valid_size):
            idx = index == img_id
            if idx.any().item():
                boxes_i = boxes[idx]
                boxes_i[:, :, [0, 2]] = torch.clamp(boxes_i[:, :, [0, 2]], min=0, max=valid_size_i[0])
                boxes_i[:, :, [1, 3]] = torch.clamp(boxes_i[:, :, [1, 3]], min=0, max=valid_size_i[1])

                boxes_out.append(boxes_i)
                scores_out.append(scores[idx])
            # else:
            #     boxes.append(None)
            #     scores.append(None)

        # return torch.stack(boxes_out, dim=0), torch.stack(scores_out, dim=0)
        return boxes_out, scores_out

    def _target_level(self, boxes:torch.Tensor) -> torch.Tensor:
        scales = (boxes[:, 2:] - boxes[:, :2]).prod(dim=-1).sqrt()
        target_level = torch.floor(self.canonical_level + torch.log2(scales / self.canonical_scale + 1e-6))
        return target_level.clamp(min=self.min_level, max=self.min_level + self.levels - 1)

    # def _rois(self, x, proposals, proposals_idx, img_size):
    #     stride = proposals.new([fs / os for fs, os in zip(x.shape[-2:], img_size)])
    #     proposals = (proposals - 0.5) * stride.repeat(2) + 0.5
    #     return roi_sampling(x, proposals, proposals_idx, self.roi_size)

    def _head_bbx(self, x:List[torch.Tensor], proposals:torch.Tensor, proposals_idx:torch.Tensor, img_size:torch.Tensor):
        # Find target levels
        target_level = self._target_level(proposals)

        # Sample rois
        rois = x[0].new_zeros(proposals.size(0), x[0].size(1), self.roi_size[0], self.roi_size[1])
        for level_i, x_i in enumerate(x):
            idx = target_level == (level_i + self.min_level)
            if idx.any().item():
                # rois[idx] = self._rois(x_i, proposals[idx], proposals_idx[idx], img_size)
                rois[idx] = g_rois(x_i, proposals[idx], proposals_idx[idx], img_size, self.roi_size)

        # Run head
        # This is to prevent batch norm from crashing when there is only ony proposal.
        prune = False
        if rois.shape[0] == 1:
            prune = True
            rois = torch.cat([rois, rois], dim=0)

        cls_logits, bbx_logits = self.head_bbx(rois)
        if prune:
            cls_logits = cls_logits[0, ...].unsqueeze(0)
            bbx_logits = bbx_logits[0, ...].unsqueeze(0)
        return cls_logits, bbx_logits

    def _head_msk(self, x:List[torch.Tensor], proposals:torch.Tensor, proposals_idx:torch.Tensor, img_size:torch.Tensor):
        # Find target levels
        target_level = self._target_level(proposals)

        # Sample rois
        rois = x[0].new_zeros(proposals.size(0), x[0].size(1), self.roi_size[0], self.roi_size[1])
        for level_i, x_i in enumerate(x):
            idx = target_level == (level_i + self.min_level)
            if idx.any().item():
                # rois[idx] = self._rois(x_i, proposals[idx], proposals_idx[idx], img_size)
                rois[idx] = g_rois(x_i, proposals[idx], proposals_idx[idx], img_size, self.roi_size)

        # Run head
        # This is to prevent batch norm from crashing when there is only ony proposal.
        prune = False
        if rois.shape[0] == 1:
            prune = True
            rois = torch.cat([rois, rois], dim=0)

        msk_logits = self.head_msk(rois)
        if prune:
            msk_logits = msk_logits[0, ...].unsqueeze(0)
        return msk_logits

    def _get_bbox_idxs(self, bbx:torch.Tensor, bbx_gt:torch.Tensor)->torch.Tensor:
        bbx_gt_idxs = torch.zeros(bbx_gt.shape[0], dtype=torch.int64, device=bbx[0].device)
        start = 0
        for idx, bbx_i in enumerate(bbx):
            bbx_gt_idxs[start:start+bbx_i.shape[0]] = idx
            start += bbx_i.shape[0]
        return bbx_gt_idxs

    def _make_batch_list(self, msk_gt_logits:torch.Tensor, bbx_gt_idx:torch.Tensor, batch_size:int):
        msk_gt_list = []
        unique_idxs = torch.unique(bbx_gt_idx)
        for entry in range(batch_size):
            if torch.sum(unique_idxs == entry) == 0:
                # msk_gt_list.append(None)
                pass
            else:
                mask = (bbx_gt_idx == entry)
                if msk_gt_logits is not None:
                    msk_gt_list.append(msk_gt_logits[mask, ...])

        return msk_gt_list

    def return_empty(self):
        bbx_empty = torch.empty([0, 4], dtype=torch.float)
        cls_empty = torch.empty([0], dtype = torch.int)
        obj_empty = torch.empty([0], dtype = torch.float)
        roi_msk_empty = torch.empty([0, 4, 28, 28], dtype=torch.float)
        return [bbx_empty], [cls_empty], [obj_empty], [roi_msk_empty]

    def forward(self, x:List[torch.Tensor], proposals:List[torch.Tensor], valid_size:List[torch.Tensor], img_size:torch.Tensor):
        x = x[self.min_level:self.min_level + self.levels]
        bbx_pred = []
        cls_pred = []
        obj_pred = []
        msk_logits = []

        if len(proposals) < 1:
            # return bbx_pred, cls_pred, obj_pred, msk_logits
            return self.return_empty()

        # proposals_2 = torch.cat(proposals)
        # logger.debug("1- proposals: {}".format(proposals_2.shape))
        # Run head on the given proposals
        # proposals = PackedSequence(proposals)
        # proposals, proposals_idx = proposals.contiguous
        proposals = torch.stack(proposals, dim=0)
        proposals, proposals_idx = g_packed_sequence_contiguous(proposals)
        # logger.debug("2- proposals: {}, idx: {}".format(proposals.shape, proposals_idx))
        # logger.debug("InstanceSegAlgoFPN, proposals: {}".format(proposals.shape))
        cls_logits, bbx_logits = self._head_bbx(x, proposals, proposals_idx, img_size)
        # logger.debug("InstanceSegAlgoFPN, FPNMaskHead-1, cls_logits: {}, bbx_logits: {}".format(cls_logits.shape, bbx_logits.shape))
        
        # Shift the proposals according to the logits
        # bbx_reg_weights = x[0].new(self.bbx_reg_weights)
        bbx_reg_weights = torch.tensor(self.bbx_reg_weights).to(x[0].device)

        boxes = g_shift_boxes(proposals.unsqueeze(1), bbx_logits / bbx_reg_weights)
        scores = torch.softmax(cls_logits, dim=1)

        # Split boxes and scores by image, clip to valid size
        boxes, scores = self._split_and_clip(boxes, scores, proposals_idx, valid_size)
        if len(boxes) < 1 or len(scores) < 1:
            # return bbx_pred, cls_pred, obj_pred, msk_logits
            return self.return_empty()
        boxes = torch.stack(boxes, dim=0)
        scores = torch.stack(scores, dim=0)
        # Do nms to find final predictions
        # logger.debug("g_prediction_generator input: boxes: {}, scores: {}".format(boxes.shape, scores.shape))
        # logger.debug("g_prediction_generator code: {}".format(g_prediction_generator.code))
        bbx_pred, cls_pred, obj_pred = g_prediction_generator(boxes, scores, self.nms_threshold, self.score_threshold, self.max_predictions)
        # logger.debug("PredictionGenerator, bbx_pred: {}, cls_pred: {}, obj_pred: {}".format(bbx_pred, cls_pred, obj_pred))

        if len(bbx_pred) < 1:
            # return bbx_pred, cls_pred, obj_pred, msk_logits
            return self.return_empty()

        # bbx_pred = PackedSequence(bbx_pred)
        # cls_pred = PackedSequence(cls_pred)
        # obj_pred = PackedSequence(obj_pred)
        # Run head again on the finalized boxes to compute instance masks
        # proposals, proposals_idx = bbx_pred.contiguous
        proposals, proposals_idx = g_packed_sequence_contiguous(torch.stack(bbx_pred, dim=0))
        # logger.debug("3- proposals: {}, idx: {}".format(proposals.shape, proposals_idx))
        # proposals = torch.cat(bbx_pred, dim=0)
        # proposals_idx 
        msk_logits = self._head_msk(x, proposals, proposals_idx, img_size)
        # logger.debug("InstanceSegAlgoFPN, FPNMaskHead-2, proposals: {}, proposals_idx: {}, msk_logits: {}".format(proposals.shape, proposals_idx.shape, msk_logits.shape))

        # Finalize instance mask computation
        batch_size = len(bbx_pred)
        # msk_pred = self.msk_prediction_generator(cls_pred, msk_logits)
        msk_logits = self._make_batch_list( msk_logits, proposals_idx, batch_size)

        # bbx_pred = torch.stack(bbx_pred, dim=0)
        # cls_pred = torch.stack(cls_pred, dim=0)
        # obj_pred = torch.stack(obj_pred, dim=0)
        # msk_logits = torch.stack(msk_logits, dim=0)

        return bbx_pred, cls_pred, obj_pred, msk_logits
