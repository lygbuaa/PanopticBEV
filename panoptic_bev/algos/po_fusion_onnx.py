import torch
import torch.nn.functional as F
from panoptic_bev.utils.parallel import PackedSequence
from panoptic_bev.utils import plogging
logger = plogging.get_logger()

@torch.jit.script
def po_process_semantic_logits(sem_logits:torch.Tensor, boxes:torch.Tensor, classes:torch.Tensor, po_num_stuff:int=6, po_num_thing:int=4, po_sem_stride:int=1):
    po_logits_stuff, po_logits_inst = [], []

    # Handle features from the semantic head
    for sem_logits_i, bbx_i, cat_i in zip(sem_logits, boxes, classes):
        # Handle stuff
        po_logits_stuff_i = sem_logits_i[:po_num_stuff, ...]

        # Handle thing
        if (bbx_i is None) or (cat_i is None):
            po_logits_stuff.append(po_logits_stuff_i)
            po_logits_inst.append(None)
        else:
            bbx_i = bbx_i / po_sem_stride

            po_logits_inst_i = torch.ones((cat_i.shape[0], po_logits_stuff_i.shape[1], po_logits_stuff_i.shape[2]),
                                            device=sem_logits.device) * -100
            for box_id in range(cat_i.shape[0]):
                # y_min = int(bbx_i[box_id][0])
                # y_max = int(bbx_i[box_id][2].round() + 1)
                # x_min = int(bbx_i[box_id][1])
                # x_max = int(bbx_i[box_id][3].round() + 1)
                y_min = bbx_i[box_id][0].to(torch.long)
                y_max = (bbx_i[box_id][2].round() + 1).to(torch.long)
                x_min = bbx_i[box_id][1].to(torch.long)
                x_max = (bbx_i[box_id][3].round() + 1).to(torch.long)

                po_logits_inst_i[box_id, y_min:y_max, x_min:x_max] = \
                    sem_logits_i[cat_i[box_id] + po_num_stuff, y_min:y_max, x_min:x_max]

            po_logits_stuff.append(po_logits_stuff_i)
            po_logits_inst.append(po_logits_inst_i)

    return po_logits_stuff, po_logits_inst

@torch.jit.script
def po_process_mask_logits(sem_logits:torch.Tensor, roi_msk_logits:torch.Tensor, boxes:torch.Tensor, classes:torch.Tensor, img_size:torch.Tensor):
    po_logits_mask = []

    # Handle features from the instance head
    for sem_logits_i, masks_i, bbx_i, cat_i in zip(sem_logits, roi_msk_logits, boxes, classes):
        if (bbx_i is None) or (cat_i is None):
            po_logits_mask.append(None)
        else:
            po_logits_mask_i = torch.ones((cat_i.shape[0], sem_logits_i.shape[1], sem_logits_i.shape[2]),
                                            device=sem_logits.device) * -100

            for box_id in range(cat_i.shape[0]):
                ref_box = bbx_i[box_id, :].long()
                y_min = bbx_i[box_id][0].to(torch.long)
                y_max = bbx_i[box_id][2].to(torch.long)
                x_min = bbx_i[box_id][1].to(torch.long)
                x_max = bbx_i[box_id][3].to(torch.long)

                w = max((x_max - x_min + 1), 1)
                h = max((y_max - y_min + 1), 1)

                roi_edge = masks_i.shape[2]
                mask = F.upsample(masks_i[box_id, :, :].view(1, 1, roi_edge, roi_edge), size=(h, w),
                                    mode="bilinear", align_corners=False).squeeze(0)
                x_min = max(ref_box[1], 0)
                x_max = min(ref_box[3] + 1, img_size[1].to(ref_box.device))
                y_min = max(ref_box[0], 0)
                y_max = min(ref_box[2] + 1, img_size[0].to(ref_box.device))

                po_logits_mask_i[box_id, y_min:y_max, x_min:x_max] = \
                    mask[0, (y_min - ref_box[0]):(y_max - ref_box[0]), (x_min - ref_box[1]):(x_max - ref_box[1])]

            po_logits_mask.append(po_logits_mask_i)

    return po_logits_mask

@torch.jit.script
def po_assign_class_label(po_pred:torch.Tensor, sem_logits:torch.Tensor, cls:torch.Tensor, po_min_stuff_area:int=0, po_num_stuff:int=6):
    po_2ch = []
    for po_pred_i, sem_logits_i, cls_i in zip(po_pred, sem_logits, cls):
        sem_pred_i = torch.max(torch.softmax(sem_logits_i, dim=0), dim=0)[1]
        po_sem_i = po_pred_i.clone()
        po_inst_i = po_pred_i.clone()

        ids = torch.unique(po_pred_i)
        ids_inst = ids[ids >= po_num_stuff]
        po_inst_i[po_inst_i < po_num_stuff] = -1

        if cls_i is not None:
            for idx, inst_id in enumerate(ids_inst):
                region = (po_inst_i == inst_id)
                # if inst_id == 255:
                #     po_sem_i[region] = 255
                #     po_inst_i[region] = -1
                #     continue

                # Get the different semantic class IDs in the instance region
                sem_cls_i, sem_cnt_i = torch.unique(sem_pred_i[region], return_counts=True)
                sem_cnt_i_max = torch.argmax(sem_cnt_i.to(torch.int32))

                tmp_cls = int(cls_i[inst_id - po_num_stuff]) + po_num_stuff

                # po_num_stuff = torch.tensor(6, dtype=torch.long, device=po_pred.device)
                if sem_cls_i[sem_cnt_i_max] == cls_i[inst_id - po_num_stuff] + po_num_stuff:
                    # The semantic and instance class IDs agree with each other.
                    ### lead to ONNXRuntimeError: 'ScatterND_587' updates tensor should have shape
                    # po_sem_i[region] = cls_i[inst_id - po_num_stuff] + po_num_stuff
                    po_sem_i[region] = tmp_cls
                    po_inst_i[region] = idx
                    # print("po_assign_class_label-1, region: {}, sem_cls_i: {}, sem_cnt_i_max: {}, cls_i: {}, cls_i_idx: {}, po_sem_i: {}".format(region.shape, sem_cls_i.shape, sem_cnt_i_max, cls_i.shape, inst_id - po_num_stuff, po_sem_i.shape))
                else:
                    # The semantic and instance class IDs do not agree with each other
                    ### torch.rand tensor only rush into this branch ###
                    if (torch.max(sem_cnt_i).type(torch.float) / torch.sum(sem_cnt_i).type(torch.float) >= 0.5) \
                            and (sem_cls_i[sem_cnt_i_max] < po_num_stuff):
                        # If the frequency of the mode is more than 0.5 and the sem label is a "stuff",
                        # assign the stuff label to it
                        po_sem_i[region] = sem_cls_i[sem_cnt_i_max]
                        po_inst_i[region] = -1
                        # print("po_assign_class_label-2, region: {}, sem_cls_i: {}, sem_cnt_i_max: {}, cls_i: {}, cls_i_idx: {}, po_sem_i: {}".format(region.shape, sem_cls_i.shape, sem_cnt_i_max, cls_i.shape, inst_id - po_num_stuff, po_sem_i.shape))
                    else:
                        # Else assign the instance segmentation class label to it
                        # po_sem_i[region] = cls_i[inst_id - po_num_stuff] + po_num_stuff
                        po_sem_i[region] = tmp_cls
                        po_inst_i[region] = idx
                        # print("po_assign_class_label-3, region: {}, sem_cls_i: {}, sem_cnt_i_max: {}, cls_i: {}, cls_i_idx: {}, po_sem_i: {}".format(region.shape, sem_cls_i.shape, sem_cnt_i_max, cls_i.shape, inst_id - po_num_stuff, po_sem_i.shape))

            idx_sem = torch.unique(po_sem_i)
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] < po_num_stuff:
                    region = (po_sem_i == idx_sem[i])
                    if region.sum() < po_min_stuff_area:
                        po_sem_i[region] = 255

        po_2ch_i = torch.zeros((po_pred_i.shape[0], po_pred_i.shape[1], 2), dtype=torch.int)
        po_2ch_i[:, :, 0] = po_sem_i
        po_2ch_i[:, :, 1] = po_inst_i
        po_2ch.append(po_2ch_i)

    return po_2ch

@torch.jit.script
def po_generate_seamless_output(po_2ch:torch.Tensor, po_num_stuff:int=6):
    po_cls = []
    po_pred_seamless = []
    po_iscrowd = []

    for po_2ch_i in po_2ch:
        po_sem_i = po_2ch_i[:, :, 0].to(torch.int64)
        po_inst_i = po_2ch_i[:, :, 1].to(torch.int64)

        # Generate seamless-style panoptic output
        # po_cls_i = [255]
        po_cls_i = torch.tensor([255], dtype=torch.int64)
        po_pred_seamless_i = torch.zeros_like(po_sem_i, dtype=torch.long)

        # Handle stuff
        classes = torch.unique(po_sem_i)
        stuff_classes = classes[classes < po_num_stuff]
        # print("stuff_classes: {}".format(stuff_classes.shape))
        for idx, cls in enumerate(stuff_classes):
            region = (po_sem_i == cls)
            # po_pred_seamless_i[region] = len(po_cls_i)  # Give the new index, but const=1 in onnx
            po_pred_seamless_i[region] = idx+1 # idx+1 will assign correct stuff class
            # po_cls_i.append(cls.item())
            po_cls_i = torch.cat((po_cls_i, cls.unsqueeze(0)), dim=0)

        # Handle instances
        instances = torch.unique(po_inst_i)
        valid_instances = instances[instances >= 0]
        for idx, inst_id in enumerate(valid_instances):
            region = (po_inst_i == inst_id)
            # po_pred_seamless_i[region] = len(po_cls_i) # Give the new index, but const=1 in onnx
            po_pred_seamless_i[region] = po_cls_i.size(dim=0)
            # po_cls_i.append(torch.unique(po_sem_i[region])[0].item())
            po_cls_i = torch.cat((po_cls_i, torch.unique(po_sem_i[region])[0].unsqueeze(0)), dim=0)
        # po_iscrowd_i = [0] * len(po_cls_i)
        po_iscrowd_i = torch.tensor([0], dtype=torch.int64).expand(po_cls_i.size(dim=0))

        po_pred_seamless.append(po_pred_seamless_i)
        # po_cls.append(torch.tensor(po_cls_i))
        po_cls.append(po_cls_i)
        po_iscrowd.append(po_iscrowd_i)

    return po_pred_seamless, po_cls, po_iscrowd

class Pofusion_ONNX(torch.nn.Module):
    def __init__(self, img_size):
        super(Pofusion_ONNX, self).__init__()
        self.img_size = img_size

    def forward(self, sem_logits:torch.Tensor, roi_msk_logits:torch.Tensor, bbx:torch.Tensor, cls:torch.Tensor):
        # During inference, cls has instance classes starting from 0, i.e, from [0, num_thing)
        # Get the roi mask containing the GT
        msk_logits = []
        for roi_msk_logits_i, cls_i in zip(roi_msk_logits, cls):
            if roi_msk_logits_i is None:
                msk_logits.append(None)
            else:
                msk_logits_i = torch.cat([roi_msk_logits_i[idx, cls_i[idx], :, :].unsqueeze(0) for idx in range(roi_msk_logits_i.shape[0])], dim=0)
                msk_logits.append(msk_logits_i)

        msk_logits = torch.stack(msk_logits, dim=0)
        po_logits_stuff, po_logits_inst = po_process_semantic_logits(sem_logits, bbx, cls)
        po_logits_mask = po_process_mask_logits(sem_logits, msk_logits, bbx, cls, self.img_size)

        po_pred = []
        # po_logits = []
        for stuff_i, inst_i, mask_i in zip(po_logits_stuff, po_logits_inst, po_logits_mask):
            if (inst_i is None) or (mask_i is None):
                po_logits_i = stuff_i
            else:
                combined_inst_i = (inst_i.sigmoid() + mask_i.sigmoid()) * (inst_i + mask_i)
                po_logits_i = torch.cat([stuff_i, combined_inst_i], dim=0)
            # po_logits.append(po_logits_i)
            po_pred_i = torch.max(torch.softmax(po_logits_i, dim=0), dim=0)[1]
            po_pred.append(po_pred_i)
        po_pred = torch.stack(po_pred, dim=0)

        # Get the panoptic instance labels for every pixel.
        # There could be some discrepancy between the class predicted by semantic seg and instance seg
        po_2ch = po_assign_class_label(po_pred, sem_logits, cls)
        po_2ch = torch.stack(po_2ch, dim=0)

        po_pred_seamless, po_cls, po_iscrowd = po_generate_seamless_output(po_2ch)
        po_pred_seamless = torch.cat(po_pred_seamless, dim=0)
        po_cls = torch.cat(po_cls, dim=0)
        po_iscrowd = torch.cat(po_iscrowd, dim=0)

        return po_pred_seamless, po_cls, po_iscrowd
