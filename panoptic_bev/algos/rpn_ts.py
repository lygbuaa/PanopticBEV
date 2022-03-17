import torch, math, sys
import torch.nn.functional as F
from panoptic_bev.utils.parallel import PackedSequence
from panoptic_bev.utils.bbx import shift_boxes
from panoptic_bev.utils import plogging
logger = plogging.get_logger()

@torch.jit.script
def g_proposal_generator(boxes:torch.Tensor, scores:torch.Tensor, nms_threshold:float=0.7, num_pre_nms:int=6000, num_post_nms:int=300, min_size:int=0):
    """Perform NMS-based selection of proposals

    Parameters
    ----------
    boxes : torch.Tensor
        Tensor of bounding boxes with shape N x M
    scores : torch.Tensor
        Tensor of bounding box scores with shape N x M x 4
    training : bool
        Switch between training and validation modes

    Returns
    -------
    proposals : PackedSequence
        Sequence of N tensors of selected bounding boxes with shape M_i x 4, entries can be None
    """

    proposals = []
    for bbx_i, obj_i in zip(boxes, scores):
        # Score pre-selection
        obj_i, idx = obj_i.topk(min(obj_i.size(0), num_pre_nms))
        bbx_i = bbx_i[idx]

        # NMS
        # idx = nms(bbx_i, obj_i, self.nms_threshold, num_post_nms)
        idx = torch.ops.po_cpp_ops.po_nms(bbx_i, obj_i, nms_threshold, num_post_nms)
        # logger.debug("ProposalGenerator bbx_i: {}, obj_i: {}, idx: {}, proposals len: {}".format(bbx_i.shape, obj_i.shape, idx.shape, len(proposals)))

        # if idx.numel() == 0:
        #     return [None]
        bbx_i = bbx_i[idx]
        proposals.append(bbx_i)
    return proposals
    # return torch.stack(proposals, dim=0)

class RPNAlgoFPN_JIT(torch.nn.Module):
    """RPN algorithm for FPN-based region proposal networks

    Parameters
    ----------
    proposal_generator : RPNProposalGenerator
    anchor_matcher : RPNAnchorMatcher
    loss : RPNLoss
    anchor_scale : list
        Anchor scale factor, this is multiplied by the RPN stride at each level to determine the actual anchor sizes
    anchor_ratios : sequence of float
        Anchor aspect ratios
    anchor_strides: sequence of int
        Effective strides of the RPN outputs at each FPN level
    min_level : int
        First FPN level to work on
    levels : int
        Number of FPN levels to work on
    """

    def __init__(self, head,
                 nms_threshold, 
                 num_pre_nms, 
                 num_post_nms, 
                 min_size,
                 anchor_scales,
                 anchor_ratios,
                 anchor_strides,
                 min_level,
                 levels):
        # super(RPNAlgoFPN_JIT, self).__init__(anchor_scale, anchor_ratios)
        super(RPNAlgoFPN_JIT, self).__init__()
        self.head = head
        self.min_level = min_level
        self.levels = levels
        self.nms_threshold = nms_threshold
        self.num_pre_nms = num_pre_nms
        self.num_post_nms = num_post_nms
        self.min_size = min_size
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        # Cache per-cell anchors
        self.anchor_strides = anchor_strides[min_level:min_level + levels]
        self.anchors = [self._base_anchors(stride) for stride in self.anchor_strides]

    def set_head(self, head):
        self.head = head
        # self.head_jit = torch.jit.load("../jit/rpn_head.pt")

    def _base_anchors(self, stride):
        # Pre-generate per-cell anchors
        anchors = []
        center = stride / 2.
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                h = stride * scale * math.sqrt(ratio)
                w = stride * scale * math.sqrt(1. / ratio)

                anchor = (
                    center - h / 2.,
                    center - w / 2.,
                    center + h / 2.,
                    center + w / 2.
                )
                anchors.append(anchor)

        return anchors

    @staticmethod
    def _shifted_anchors(anchors, stride, height, width, dtype=torch.float32, device="cpu"):
        grid_y = torch.arange(0, stride * height, stride, dtype=dtype, device=device)
        grid_x = torch.arange(0, stride * width, stride, dtype=dtype, device=device)
        grid = torch.stack([grid_y.view(-1, 1).repeat(1, width), grid_x.view(1, -1).repeat(height, 1)], dim=-1)

        anchors = torch.tensor(anchors, dtype=dtype, device=device)
        shifted_anchors = anchors.view(1, 1, -1, 4) + grid.repeat(1, 1, 2).unsqueeze(2)
        return shifted_anchors.view(-1, 4)

    def _get_logits(self, x):
        obj_logits, bbx_logits, h, w = [], [], [], []
        for x_i in x:
            obj_logits_i, bbx_logits_i = self.head(x_i)
            logger.debug("x_i: {}, obj_logits_i: {}, bbx_logits_i:{}".format(x_i.shape, obj_logits_i.shape, bbx_logits_i.shape))
            # remove int(s) shall be safe
            h_i, w_i = (s for s in obj_logits_i.shape[-2:])
            # print("h_i: {}, w_i: {}".format(h_i, w_i))

            obj_logits_i = obj_logits_i.permute(0, 2, 3, 1).contiguous().view(obj_logits_i.size(0), -1)
            bbx_logits_i = bbx_logits_i.permute(0, 2, 3, 1).contiguous().view(bbx_logits_i.size(0), -1, 4)

            obj_logits.append(obj_logits_i)
            bbx_logits.append(bbx_logits_i)
            h.append(h_i)
            w.append(w_i)

        return torch.cat(obj_logits, dim=1), torch.cat(bbx_logits, dim=1), h, w

    def _inference(self, obj_logits, bbx_logits, anchors, valid_size):
        # Compute shifted boxes
        boxes = shift_boxes(anchors, bbx_logits)
        logger.debug("RPNAlgoFPN anchors: {}, bbx_logits: {}, boxes: {}".format(anchors.shape, bbx_logits.shape, boxes.shape))

        # Clip boxes to their image sizes, valid_size is one-tensor-list: [torch.Size([W, Z])]
        for i, (height, width) in enumerate(valid_size):
            boxes[i, :, [0, 2]] = boxes[i, :, [0, 2]].clamp(min=0, max=height)
            boxes[i, :, [1, 3]] = boxes[i, :, [1, 3]].clamp(min=0, max=width)

        return g_proposal_generator(boxes, obj_logits, self.nms_threshold, self.num_pre_nms, self.num_post_nms, self.min_size)

    def forward(self, x, valid_size):
        # Calculate logits for the levels that we need
        x = x[self.min_level:self.min_level + self.levels]
        obj_logits, bbx_logits, h, w = self._get_logits(x)
        # these tensor stay the same every iteration
        # logger.debug("anchors: {}, anchor_strides: {}, h: {}, w: {}, valid_size: {}".format(self.anchors, self.anchor_strides, h, w, valid_size))

        # Compute anchors for each scale and merge them
        anchors = []
        for h_i, w_i, stride_i, anchors_i in zip(h, w, self.anchor_strides, self.anchors):
            anchors.append(self._shifted_anchors(anchors_i, stride_i, h_i, w_i, bbx_logits.dtype, bbx_logits.device))
        anchors = torch.cat(anchors, dim=0)

        return self._inference(obj_logits, bbx_logits, anchors, valid_size)
