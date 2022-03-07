#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import torch
import math, os
from collections import OrderedDict
import inplace_abn
from panoptic_bev.utils import plogging
# plogging.init("./", "fuse_conv_bn") #remember, in this project, plogging.init() no more than once
logger = plogging.get_logger()

# load images, intrinsics, extrinsics
class FuseConvBn(object):
    g_layers = []
    g_conv_bn_pairs = []
    g_last_conv = None

    def __init__(self):
        pass

    def print_layers(self, model):
        for idx, (path, submodule) in enumerate(model.named_modules()):
            logger.info("named_modules-{} - {} - {}: {}".format(idx, path, submodule.__class__.__name__, submodule))

    def enumerate(self, model):
        for n, module in model.named_children():
            n_children = len(list(module.children()))
            logger.info("module_path:{}-{}, type: {}, children {}".format(model.__class__.__name__, n, module.__class__.__name__, n_children))
            # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ConvTranspose2d):
            #     logger.info("module weight: {}".format(module.state_dict()["weight"]))
            if n_children > 0:
                ## compound module, go inside it
                self.enumerate(module)
            else:
                logger.info("module {} dict: {}".format(module, module.state_dict()))

    def replace_layer(self, model, old, new):
        for n, module in model.named_children():
            n_children = len(list(module.children()))
            logger.info("module_path:{}-{}, type: {}, children {}".format(model.__class__.__name__, n, module.__class__.__name__, n_children))
            if n_children > 0:
                ## compound module, go inside it
                self.replace_layer(module, old, new)

            if isinstance(module, old):
                # logger.info("module dict: {}".format(module.state_dict()))
                setattr(model, n, new)
                confirm = getattr(model, n)
                logger.info("replace {}-{}-{} to {}".format(model.__class__.__name__, n, module.__class__.__name__, confirm))

    def extract_bns(self, model, bn_type=inplace_abn.abn.InPlaceABNSync):
        for n, module in model.named_children():
            n_children = len(list(module.children()))
            logger.info("module_path:{}-{}, type: {}, children {}".format(model.__class__.__name__, n, module.__class__.__name__, n_children))
            if n_children > 0:
                ## compound module, go inside it
                self.extract_bns(module, bn_type)
            elif isinstance(module, bn_type):
                self.g_conv_bn_pairs.append((self.g_last_conv, module))
                old_W = self.g_last_conv.state_dict()["weight"]
                new_W = torch.zeros(old_W.size()).float().to(old_W.device)
                self.g_last_conv.weight.data.copy_(new_W)
                logger.info("conv-{} weight covered!".format(self.g_last_conv))
            else:
                self.g_last_conv = module

    def get_conv_bn_pairs(self):
        logger.info("g_conv_bn_pairs total: {}".format(len(self.g_conv_bn_pairs)))

        for pairs in self.g_conv_bn_pairs:
            logger.info("conv: {}-{}, bn: {}-{}, eps: {}".format(pairs[0], pairs[0].state_dict().keys(), pairs[1], pairs[1].state_dict().keys(), pairs[1].eps))

        return self.g_conv_bn_pairs

    def fuse_bn_to_conv(self, bn_layer, conv_layer):
        bn_st_dict = bn_layer.state_dict()
        conv_st_dict = conv_layer.state_dict()
        # logger.info("bn_st_dict: {}, conv_st_dict: {}".format(bn_st_dict, conv_st_dict))
        
        # BatchNorm params
        eps = bn_layer.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv_layer.weight.data.copy_(W)
        if conv_layer.bias is None:
            conv_layer.bias = torch.nn.Parameter(bias)
        else:
            conv_layer.bias.data.copy_(bias)

    def do_bn_fusion(self, model, bn_type=inplace_abn.abn.InPlaceABNSync):
        for n, module in model.named_children():
            n_children = len(list(module.children()))
            logger.info("module_path:{}-{}, type: {}, children {}".format(model.__class__.__name__, n, module.__class__.__name__, n_children))
            if n_children > 0:
                ## compound module, go inside it
                self.do_bn_fusion(module, bn_type)
            elif isinstance(module, bn_type):
                # self.g_conv_bn_pairs.append((self.g_last_conv, module))
                # old_W = self.g_last_conv.state_dict()["weight"]
                # new_W = torch.zeros(old_W.size()).float().to(old_W.device)
                # self.g_last_conv.weight.data.copy_(new_W)
                # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ConvTranspose2d):
                if isinstance(self.g_last_conv, torch.nn.Conv2d) or isinstance(self.g_last_conv, torch.nn.Conv3d) or isinstance(self.g_last_conv, torch.nn.Linear) or isinstance(self.g_last_conv, torch.nn.ConvTranspose2d):
                    self.fuse_bn_to_conv(module, self.g_last_conv)
                    logger.info("fuse conv-{} && bn-{}".format(self.g_last_conv, module))
                    setattr(model, n, torch.nn.Identity())
                    # logger.info("removed {}-{}-{}".format(model.__class__.__name__, n, module.__class__.__name__))
            else:
                self.g_last_conv = module

    def make_model_demo(self):
        encoder = torch.nn.Sequential(OrderedDict([
            ("conv1", torch.nn.Conv2d(3, 64, 5)),
            ("bn1", torch.nn.SyncBatchNorm(100)),
            ("relu1", torch.nn.ReLU()),
            ("conv2", torch.nn.Conv2d(64, 64, 5)),
            ("bn2", torch.nn.SyncBatchNorm(100)),
            ("relu2", torch.nn.ReLU())
        ]))

        decoder = torch.nn.Sequential(OrderedDict([
            ("conv1", torch.nn.Conv2d(64, 3, 5)),
            ("bn1", inplace_abn.abn.InPlaceABNSync(100)),
            ("conv2", torch.nn.Conv2d(3, 1, 5)),
            ("bn2", inplace_abn.abn.InPlaceABNSync(100)),
        ]))

        model = torch.nn.Sequential(OrderedDict([
            ("encoder", encoder),
            ("decoder", decoder)
        ]))
        return model

if __name__ == '__main__':
    fuser = FuseConvBn()
    model = fuser.make_model_demo()
    fuser.enumerate(model)
    # print(model)
    # fuser.replace_layer(model, torch.nn.SyncBatchNorm, torch.nn.Identity())
    # print(model)

    # self.replace_layer(model, inplace_abn.abn.InPlaceABNSync, torch.nn.Identity())
    # self.enumerate(model)