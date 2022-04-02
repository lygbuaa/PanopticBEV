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
            if isinstance(module, torch.nn.SyncBatchNorm) or isinstance(module, inplace_abn.abn.InPlaceABNSync):
                weight = module.state_dict()["weight"]
                bias = module.state_dict()["bias"]
                running_mean = module.state_dict()["running_mean"]
                running_var = module.state_dict()["running_var"]
                logger.info("module {}, weight: {}, bias: {}, running_mean: {}, running_var: {}".format(module, weight.shape, bias.shape, running_mean.shape, running_var.shape))
            if n_children > 0:
                ## compound module, go inside it
                self.enumerate(module)
            else:
                pass
                # logger.info("module {} dict: {}".format(module, module.state_dict()))

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

    # fuse torch.nn.SyncBatchNorm to conv
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

    # v1 sometimes insert Identity() fails
    def do_bn_fusion_v1(self, model, bn_type=inplace_abn.abn.InPlaceABNSync):
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
                if isinstance(self.g_last_conv, torch.nn.Conv2d): # or isinstance(self.g_last_conv, torch.nn.Conv3d) or isinstance(self.g_last_conv, torch.nn.Linear) or isinstance(self.g_last_conv, torch.nn.ConvTranspose2d):
                    self.fuse_bn_to_conv(module, self.g_last_conv)
                    logger.info("fuse conv-{} && bn-{}".format(self.g_last_conv, module))
                    setattr(model, n, torch.nn.Identity())
                    # logger.info("removed {}-{}-{}".format(model.__class__.__name__, n, module.__class__.__name__))
            else:
                self.g_last_conv = module

    def extract_layers(self, model):
        list_layers = []
        for n, p in model.named_modules():
            list_layers.append(n)
        # logger.info("list_layers: {}".format(list_layers))
        return list_layers

    def extract_layer(self, model, layer):
        layer = layer.split('.')
        module = model
        if hasattr(model, 'module') and layer[0] != 'module':
            module = model.module
        if not hasattr(model, 'module') and layer[0] == 'module':
            layer = layer[1:]
        for l in layer:
            if hasattr(module, l):
                if not l.isdigit():
                    module = getattr(module, l)
                else:
                    module = module[int(l)]
            else:
                return module
        return module

    def set_layer(self, model, layer, val):
        layer = layer.split('.')
        module = model
        if hasattr(model, 'module') and layer[0] != 'module':
            module = model.module
        lst_index = 0
        module2 = module
        for l in layer:
            if hasattr(module2, l):
                if not l.isdigit():
                    module2 = getattr(module2, l)
                else:
                    module2 = module2[int(l)]
                lst_index += 1
        lst_index -= 1
        for l in layer[:lst_index]:
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        l = layer[lst_index]
        setattr(module, l, val)

    def compute_next_bn(self, layer_name, resnet, bn_name="SyncBatchNorm"):
        list_layer = self.extract_layers(resnet)
        assert layer_name in list_layer
        if layer_name == list_layer[-1]:
            return None
        next_bn = list_layer[list_layer.index(layer_name) + 1]
        if self.extract_layer(resnet, next_bn).__class__.__name__ == bn_name:
            return next_bn
        return None

    def make_placeholder(self, abn_layer):
        abn_st_dict = abn_layer.state_dict()
        relu_layer = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        placeholder = torch.nn.Identity()
        if 'bias' in abn_st_dict:
            beta = abn_st_dict['bias']
            dim = beta.shape[0]
            # gamma = abn_st_dict['weight']
            # logger.debug("abn weight: {}, abn bias: {}".format(gamma.shape, beta.shape))
            new_bn_layer = torch.nn.SyncBatchNorm(num_features=dim, eps=1e-05, device=beta.device)
            W = torch.ones(size=(dim,))
            new_bn_layer.weight.data.copy_(W)
            new_bn_layer.bias.data.copy_(beta)
            # logger.info("new_bn_layer: {}".format(new_bn_layer.state_dict()))
            placeholder = torch.nn.Sequential(OrderedDict([
                ("leaky_relu", relu_layer),
                ("new_bn", new_bn_layer),
            ]))        
        else:
            # logger.debug("abn bias None")
            placeholder = relu_layer
        return placeholder

    """ fuse InPlaceABNSync to conv, InPlace Activated Batch Normalization
    This applies the following per-channel combined BatchNorm + activation operation:

        x_hat = (x - mu) / sqrt(sigma^2 + eps)
        x <- act(x_hat, p) * (|weight| + eps) + bias

    where:
        - mu is the per-channel batch mean, or `running_mean` if `training` is `False`
        - sigma^2 is the per-channel batch variance, or `running_var` if `training` is `False`
        - act(., p) is the activation function specified by `activation`
        - p is `activation_param`, i.e. the negative slope of Leaky ReLU or alpha
          parameter of ELU
        - `weight` and `bias` are the optional affine parameters
        - `eps` is a small positive number
    """
    def fuse_abn_to_conv(self, abn_layer, conv_layer):
        abn_st_dict = abn_layer.state_dict()
        conv_st_dict = conv_layer.state_dict()
        # logger.info("bn_st_dict: {}, conv_st_dict: {}".format(bn_st_dict, conv_st_dict))
        
        # BatchNorm params
        eps = abn_layer.eps
        mu = abn_st_dict['running_mean']
        var = abn_st_dict['running_var']
        gamma = torch.abs(abn_st_dict['weight']) + eps
        # gamma = abn_st_dict['weight']

        # if 'bias' in abn_st_dict:
        #     beta = abn_st_dict['bias']
        # else:
        # abn bias must add after activation
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

    # v2 can replace every bn layer with placeholder
    def do_bn_fusion_v2(self, model, bn_name="SyncBatchNorm", placeholder=torch.nn.Identity()):
        bn_counter = 0
        for n, m in model.named_modules():
            #if isinstance(m, torch.nn.Conv3d): #Conv2d, Conv3d, Linear, ConvTranspose2d
                next_bn = self.compute_next_bn(n, model, bn_name)
                # logger.info("enumerate, n: {}, m: {}, next_bn: {}".format(n, m, next_bn))
                if next_bn is not None:
                    next_bn_ = self.extract_layer(model, next_bn)
                    if bn_name=="SyncBatchNorm":
                        self.fuse_bn_to_conv(next_bn_, m)
                    elif bn_name == "InPlaceABNSync":
                        self.fuse_abn_to_conv(next_bn_, m)
                        placeholder = self.make_placeholder(next_bn_)
                    self.set_layer(model, next_bn, placeholder)
                    bn_counter += 1
                    # logger.info("fuse {}".format(next_bn_))
        # logger.info("final model after fusion: {}".format(model))
        logger.info("fused {} bn layer".format(bn_counter))
        return model

    def make_model_demo(self):
        encoder = torch.nn.Sequential(OrderedDict([
            ("conv1", torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), bias=True)),
            ("bn1", torch.nn.SyncBatchNorm(num_features=64, eps=1e-05)),
            ("relu1", torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            ("conv2", torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), bias=True)),
            ("bn2", torch.nn.SyncBatchNorm(num_features=64, eps=1e-05)),
            ("relu2", torch.nn.LeakyReLU(negative_slope=0.01, inplace=True))
        ]))

        decoder = torch.nn.Sequential(OrderedDict([
            ("conv1", torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), bias=True)),
            ("bn1", inplace_abn.abn.InPlaceABNSync(num_features=64, eps=1e-05, activation="leaky_relu", activation_param=0.01)),
            ("conv2", torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3,3), bias=True)),
            ("bn2", inplace_abn.abn.InPlaceABNSync(num_features=3, eps=1e-05, activation="leaky_relu", activation_param=0.01)),
        ]))

        model = torch.nn.Sequential(OrderedDict([
            ("encoder", encoder),
            ("decoder", decoder)
        ]))
        return model

if __name__ == '__main__':
    fuser = FuseConvBn()
    model = fuser.make_model_demo()

    fuser.do_bn_fusion_v2(model, bn_name="InPlaceABNSync")
    logger.info("final model: {}".format(model))
