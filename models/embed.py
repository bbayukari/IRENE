# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.attention import Attention
import pdb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=1):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        tk_lim = config.cc_len
        num_lab = config.lab_len

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.lab_embeddings = Linear(1, config.hidden_size)  

        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+n_patches, config.hidden_size))
        self.pe_lab = nn.Parameter(torch.zeros(1, num_lab, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_lab = Dropout(config.transformer["dropout_rate"])

    def forward(self, img, lab):
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)

        if self.hybrid:
            img = self.hybrid_model(img)
        img = self.patch_embeddings(img)
        lab = self.lab_embeddings(torch.unsqueeze(lab, 2))

        img = img.flatten(2)
        img = img.transpose(-1, -2)
        img = torch.cat((cls_tokens, img), dim=1)

        embeddings = img + self.position_embeddings
        lab_embeddings = lab + self.pe_lab

        embeddings = self.dropout(embeddings)
        lab_embeddings = self.dropout_lab(lab_embeddings)
        return embeddings, lab_embeddings



