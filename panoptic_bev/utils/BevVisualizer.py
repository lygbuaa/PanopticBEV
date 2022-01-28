#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import torch, torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from panoptic_bev.utils import logging
from panoptic_bev.utils.semantic import g_semantic_names, g_semantic_colours, g_num_stuff, g_num_thing

class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class BevVisualizer(object):
    H_W_RATIO = 0.5833 # 448/768
    PLT_W = 5.0
    PLT_H = 1.75
    config = None
    output_dir = "./"
    rgb_mean = None
    rgb_std = None

    def __init__(self, config, vis_dir):
        self.config = config
        self.output_dir = vis_dir
        os.makedirs(vis_dir, exist_ok=True)
        dl_config = config['dataloader']
        front_resize = dl_config.getstruct("front_resize")
        self.H_W_RATIO = front_resize[0] / front_resize[1]
        self.PLT_H = self.PLT_W * self.H_W_RATIO
        self.rgb_mean = dl_config.getstruct("rgb_mean")
        self.rgb_std = dl_config.getstruct("rgb_std")

    def recover_image(self, img):
        img = img * img.new(self.rgb_std).view(-1, 1, 1)
        img = img + img.new(self.rgb_mean).view(-1, 1, 1)
        return img

    # just add frame border for each image
    def make_contour(self, img, colour=[0, 0, 0], double_line=True):
        h, w = img.shape[:2]
        out = img.copy()
        # Vertical lines
        out[np.arange(h), np.repeat(0, h)] = colour
        out[np.arange(h), np.repeat(w - 1, h)] = colour

        # Horizontal lines
        out[np.repeat(0, w), np.arange(w)] = colour
        out[np.repeat(h - 1, w), np.arange(w)] = colour

        if double_line:
            out[np.arange(h), np.repeat(1, h)] = colour
            out[np.arange(h), np.repeat(w - 2, h)] = colour

            # Horizontal lines
            out[np.repeat(1, w), np.arange(w)] = colour
            out[np.repeat(h - 2, w), np.arange(w)] = colour
        return out

    def convert_figure_numpy(self, figure):
        """ Convert figure to numpy image """
        figure.canvas.draw() # to avoid 'FigureCanvasTkAgg' object has no attribute 'renderer'
        figure_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        figure_np = figure_np.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return figure_np

    def plot_bev_segmentation(self, semantic_seg):
        semantic_seg[semantic_seg == 255] = 10 # trainId-255 to g_semantic_colours-10
        plot_image = g_semantic_colours[semantic_seg]
        # plot_bev_grids(plot_image)
        return plot_image

    def plot_bev(self, inputs, idx, bev_pred=None, bev_gt=None):
        logger = logging.get_logger()
        output_filename = os.path.join(self.output_dir, str(idx)) + '.png'
        fig = plt.figure(figsize=(2*self.PLT_W, 1*self.PLT_H))
        width_ratios = (self.PLT_W, self.PLT_W)
        gs = mpl.gridspec.GridSpec(1, 2, width_ratios=width_ratios)
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        # plot raw images
        denormalise_img = torchvision.transforms.Compose(
            (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),)
        )
        front_image = inputs["img"].cpu().contiguous[0]
        ax = plt.subplot(gs[0, 0])
        front_image = denormalise_img(front_image)
        plt.imshow(front_image)
        plt.axis("off")

        # plot bev results, cat-idx must be retrieved from po_class vector
        bev_seg_po = bev_pred['po_pred'].cpu().contiguous[0]
        seg_cls = bev_pred['po_class'].cpu().contiguous[0].numpy()
        bev_seg = seg_cls[bev_seg_po]
        # logger.info("seg max val {}".format(torch.max(bev_seg)))
        ax = plt.subplot(gs[0, 1])
        bev_image = self.plot_bev_segmentation(bev_seg)
        plt.imshow(self.make_contour(bev_image))
        plt.axis("off")

        plt.draw()
        figure_numpy = self.convert_figure_numpy(fig)
        Image.fromarray(figure_numpy).save(output_filename)
        logger.info("saved bev image {}".format(output_filename))


if __name__ == '__main__':
    pass