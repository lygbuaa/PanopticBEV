#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import torch, torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from panoptic_bev.utils import plogging
from panoptic_bev.utils.semantic import g_semantic_names, g_semantic_colours, g_num_stuff, g_num_thing
from datetime import datetime

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
    N = 1 # total input images

    def __init__(self, config, vis_dir, n_imgs = 1):
        self.config = config
        self.output_dir = vis_dir
        self.N = n_imgs
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

    def plot_bev(self, inputs, idx, bev_pred=None, bev_gt=None, show_po=True):
        logger = plogging.get_logger()
        now = datetime.now()
        dt_str = now.strftime("%Y%m%d-%H%M%S-")
        output_filename = os.path.join(self.output_dir, dt_str + str(idx)) + '.png'

        fig = plt.figure(figsize=(3*self.PLT_W, 3*self.PLT_H))
        width_ratios = (self.PLT_W, self.PLT_W, self.PLT_W)
        gs = mpl.gridspec.GridSpec(3, 3, width_ratios=width_ratios)
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        # plot raw images
        denormalise_img = torchvision.transforms.Compose(
            (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),)
        )
        input_images = inputs["img"].cpu().contiguous[0]
        # logger.debug("input img: {}".format(input_images.shape))

        if self.N == 1:
            ax = plt.subplot(gs[0, 1])
            front_image = denormalise_img(input_images)
            plt.imshow(front_image)
            plt.axis("off")
        elif self.N == 6:
            ax_f = plt.subplot(gs[0, 1])
            f_img = denormalise_img(input_images[0])
            plt.imshow(f_img)
            plt.annotate("FRONT", (0.0, 0.9), c='white', xycoords='axes fraction', fontsize=14)
            plt.axis("off")
            ax_fl = plt.subplot(gs[0, 0])
            fl_img = denormalise_img(input_images[1])
            plt.imshow(fl_img)
            plt.annotate("FRONT_LEFT", (0.0, 0.9), c='white', xycoords='axes fraction', fontsize=14)
            plt.axis("off")
            ax_bl = plt.subplot(gs[2, 0])
            # backward cameras should flip for human convenient
            bl_img = denormalise_img(input_images[2]).transpose(Image.FLIP_LEFT_RIGHT)
            plt.imshow(bl_img)
            plt.annotate("BACK_LEFT", (0.0, 0.9), c='white', xycoords='axes fraction', fontsize=14)
            plt.axis("off")
            ax_b = plt.subplot(gs[2, 1])
            b_img = denormalise_img(input_images[3]).transpose(Image.FLIP_LEFT_RIGHT)
            plt.imshow(b_img)
            plt.annotate("BACK", (0.0, 0.9), c='white', xycoords='axes fraction', fontsize=14)
            plt.axis("off")
            ax_br = plt.subplot(gs[2, 2])
            br_img = denormalise_img(input_images[4]).transpose(Image.FLIP_LEFT_RIGHT)
            plt.imshow(br_img)
            plt.annotate("BACK_RIGHT", (0.0, 0.9), c='white', xycoords='axes fraction', fontsize=14)
            plt.axis("off")
            ax_fr = plt.subplot(gs[0, 2])
            fr_img = denormalise_img(input_images[5])
            plt.imshow(fr_img)
            plt.annotate("FRONT_RIGHT", (0.0, 0.9), c='white', xycoords='axes fraction', fontsize=14)
            plt.axis("off")

        # plot bev results, cat-idx must be retrieved from po_class vector
        if show_po:
            bev_seg_po = bev_pred['po_pred'].cpu().contiguous[0]
            # logger.debug("bev_seg_po shape: {}".format(bev_seg_po.shape))
        else:
            bev_seg = bev_pred['sem_pred'].cpu().contiguous[0]
            # logger.debug("bev_seg shape: {}".format(bev_seg.shape))
        seg_cls = bev_pred['po_class'].cpu().contiguous[0].numpy()
        bbx_pred = bev_pred['bbx_pred'].cpu().contiguous[0]
        cls_pred = bev_pred['cls_pred'].cpu().contiguous[0]
        # logger.debug("seg_cls: {}, bbx_pred: {}".format(seg_cls.shape, bbx_pred.shape))

        if show_po:
            bev_seg = seg_cls[bev_seg_po]

        # logger.info("seg max val {}".format(torch.max(bev_seg)))
        ax = plt.subplot(gs[1, 1])
        bev_image = self.plot_bev_segmentation(bev_seg)
        # logger.debug("bev_image 1: {}".format(bev_image.shape))
        if bbx_pred is not None:
            for idx, bbx in enumerate(bbx_pred.numpy()):
                start_point = (int(bbx[1]), int(bbx[0]))
                end_point = (int(bbx[3]), int(bbx[2]))
                cls = cls_pred.numpy()[idx]
                rgb = g_semantic_colours[cls + g_num_stuff]
                bev_image = cv2.rectangle(bev_image, start_point, end_point, color=(int(rgb[0]), int(rgb[1]), int(rgb[2])), thickness=2)
        # draw ego-vehicle
        h, w, c = bev_image.shape
        vertices = np.array([[w/2+20, h/2], [w/2-20, h/2+15], [w/2-20, h/2-15]], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(bev_image, [vertices], color=(0,0,0))
    
        #set front camera to upwards, for convenient
        # bev_image = np.rot90(bev_image, k=1, axes=(0, 1))
        # logger.debug("bev_image 2: {}".format(bev_image.shape))
        plt.imshow(self.make_contour(bev_image))
        plt.axis("off")

        plt.draw()
        figure_numpy = self.convert_figure_numpy(fig)
        Image.fromarray(figure_numpy).save(output_filename)
        logger.info("saved bev image {}".format(output_filename))


if __name__ == '__main__':
    pass