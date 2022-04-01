from math import ceil
import torch
import torch.nn.functional as functional

from panoptic_bev.utils.parallel import PackedSequence
from panoptic_bev.utils.sequence import pack_padded_images
from panoptic_bev.utils import plogging
logger = plogging.get_logger()
class SemanticSegLoss:
    """Semantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, out_shape=(768, 704), ignore_index=255, ignore_labels=None,
                 bev_params=None, extrinsics=None):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.ignore_index = ignore_index
        self.ignore_labels = ignore_labels

        resolution = float(bev_params['cam_z']) / float(bev_params['f'])

        rows = torch.arange(0, out_shape[0])
        cols = torch.arange(0, out_shape[1])
        rr, cc = torch.meshgrid(rows, cols)
        idx_mesh = torch.cat([rr.unsqueeze(0), cc.unsqueeze(0)], dim=0)
        ego_position = torch.tensor([out_shape[0] // 2, 0]).view(-1, 1, 1)
        pos_mesh = idx_mesh - ego_position
        self.X, self.Z = pos_mesh[0] * resolution, pos_mesh[1] * resolution
        self.Y = abs(float(extrinsics['translation'][2]))

    def __call__(self, sem_logits, sem, weights_msk, intrinsics):
        """Compute the semantic segmentation loss

        Parameters
        ----------
        sem_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        sem_loss = []
        for i, (sem_logits_i, sem_i, wt_msk_i) in enumerate(zip(sem_logits, sem, weights_msk)):
            if self.ignore_labels is not None:
                sem_i[(sem_i == self.ignore_labels).any(-1)] = self.ignore_index  # Remap the ignore_labels to ignore_index
            sem_loss_i = functional.cross_entropy(sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0),
                                                  ignore_index=self.ignore_index, reduction="none")

            f_x = intrinsics[i][0][0]
            f_y = intrinsics[i][1][1]
            self.X = self.X.to(sem_i.device)
            self.Z = self.Z.to(sem_i.device)

            S = torch.sqrt((f_x**2 * self.Z**2) + (f_x*self.X + f_y*self.Y)**2) / (self.Z**2)
            sensitivity_map = 1 / torch.log(1 + S)
            sensitivity_map[torch.isnan(sensitivity_map)] = 0.
            # sensitivity_wt = sensitivity_map * 10
            sensitivity_wt = sensitivity_map * 10

            # Distance-based weighting
            sem_loss_i *= (1 + sensitivity_wt.to(sem_i.device).squeeze(0))

            # Multiply the instace-based weights mask
            sem_loss_i *= (wt_msk_i / 10000)

            sem_loss_i = sem_loss_i.view(-1)

            if self.ohem is not None and self.ohem != 1:
                top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                if top_k != sem_loss_i.numel():
                    sem_loss_i, _ = sem_loss_i.topk(top_k)

            sem_loss.append(sem_loss_i.mean())

        return sum(sem_loss) / len(sem_logits)


class SemanticSegAlgo(torch.nn.Module):
    """Semantic segmentation algorithm

    Parameters
    ----------
    loss : SemanticSegLoss
    num_classes : int
        Number of classes
    """

    def __init__(self, head, loss, num_classes, img_size, ignore_index=255):
        super(SemanticSegAlgo, self).__init__()
        self.loss = loss
        self.head = head
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.img_size = img_size

    @staticmethod
    def _pack_logits(sem_logits, valid_size, img_size):
        sem_logits = functional.interpolate(sem_logits, size=img_size.tolist(), mode="bilinear", align_corners=False)
        return pack_padded_images(sem_logits, valid_size)

    def _confusion_matrix(self, sem_pred, sem):
        confmat = sem[0].new_zeros(self.num_classes * self.num_classes, dtype=torch.float)

        for sem_pred_i, sem_i in zip(sem_pred, sem):
            valid = sem_i != self.ignore_index
            if valid.any():
                sem_pred_i = sem_pred_i[valid]
                sem_i = sem_i[valid]

                confmat.index_add_(0,
                                   sem_i.view(-1) * self.num_classes + sem_pred_i.view(-1),
                                   confmat.new_ones(sem_i.numel()))

        return confmat.view(self.num_classes, self.num_classes)

    # @staticmethod
    # def _logits(head, x, bbx, img_size, roi):
    #     sem_logits, sem_feat, roi_logits = head(x, bbx, img_size, roi)
    #     return sem_logits, sem_feat, roi_logits

    def forward(self, x):
        """Given input features compute semantic segmentation prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute semantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        sem_pred : PackedSequence
            A sequence of N tensors of semantic segmentations with shapes H_i x W_i
        """
        # sem_logits_, sem_feat, _ = self._logits(x, None, img_size, False)
        sem_logits_, _, _  = self.head(x, None, self.img_size, False)
        # sem_logits = self._pack_logits(sem_logits_, valid_size, img_size)
        # logger.debug("sem_head output sem_logits_: {}".format(sem_logits_.shape))

        sem_logits = functional.interpolate(sem_logits_, size=self.img_size.tolist(), mode="bilinear", align_corners=False)
        # logger.debug("interpolation output sem_logits: {}".format(sem_logits.shape))

        # sem_logits = pack_padded_images(sem_logits, valid_size)
        # sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])
        sem_pred = torch.cat([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])
        # logger.debug("sem_pred: {}".format(sem_pred.shape))
        return sem_pred, sem_logits
