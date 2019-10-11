#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from nmtlab.modules.transformer_modules import TransformerEncoderLayer
from nmtlab.modules.transformer_modules import TransformerFeedForward
from nmtlab.modules.transformer_modules import MultiHeadAttention
from nmtlab.modules.transformer_modules import residual_connect
from nmtlab.utils import OPTS


class TransformerEncoder(nn.Module):
    """
    Self-attention -> FF -> layer norm
    """

    def __init__(self, embed_layer, size, n_layers, ff_size=None, n_att_head=8, dropout_ratio=0.1, skip_connect=False):
        super(TransformerEncoder, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.embed_layer = embed_layer
        self.layer_norm = nn.LayerNorm(size)
        self.encoder_layers = nn.ModuleList()
        self.skip_connect = skip_connect
        self._rescale = 1. / math.sqrt(2)
        for _ in range(n_layers):
            layer = TransformerEncoderLayer(size, ff_size, n_att_head=n_att_head,
                                            dropout_ratio=dropout_ratio)
            self.encoder_layers.append(layer)

    def forward(self, x, mask=None):
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        first_x = x
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, mask)
            if self.skip_connect:
                x = self._rescale * (first_x + x)
        x = self.layer_norm(x)
        return x


class TransformerCrossEncoderLayer(nn.Module):

    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1, relative_pos=False):
        super(TransformerCrossEncoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.cross_attention = MultiHeadAttention(size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.ff_layer = TransformerFeedForward(size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)
        self.layer_norm3 = nn.LayerNorm(size)

    def forward(self, x, x_mask, y, y_mask):
        # Attention layer
        h1 = self.layer_norm1(x)
        h1, _ = self.attention(h1, h1, h1, mask=x_mask)
        h1 = self.dropout(h1)
        h1 = residual_connect(h1, x)
        # Cross-attention
        h2 = self.layer_norm2(h1)
        h2, _ = self.attention(h2, y, y, mask=y_mask)
        h2 = self.dropout(h2)
        h2 = residual_connect(h2, h1)
        # Feed-forward layer
        h3 = self.layer_norm3(h2)
        h3 = self.ff_layer(h3)
        h3 = self.dropout(h3)
        h3 = residual_connect(h3, h2)
        return h3


class TransformerCrossEncoder(nn.Module):
    """
    Self-attention -> cross-attenion -> FF -> layer norm
    """

    def __init__(self, embed_layer, size, n_layers, ff_size=None, n_att_head=8, dropout_ratio=0.1, skip_connect=False):
        super(TransformerCrossEncoder, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self._skip = skip_connect
        self._reslace = 1. / math.sqrt(2)
        self.embed_layer = embed_layer
        self.layer_norm = nn.LayerNorm(size)
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerCrossEncoderLayer(size, ff_size, n_att_head=n_att_head,
                                            dropout_ratio=dropout_ratio)
            self.encoder_layers.append(layer)

    def forward(self, x, x_mask, y, y_mask):
        if self.embed_layer is not None:
            x = self.embed_layer(x)
        first_x = x
        for l, layer in enumerate(self.encoder_layers):
            x = layer(x, x_mask, y, y_mask)
            if self._skip:
                x = self._reslace * (first_x + x)
        x = self.layer_norm(x)
        return x


class LengthConverter(nn.Module):
    """
    Implementation of Length Transformation.
    """

    def __init__(self):
        super(LengthConverter, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(1., dtype=torch.float))

    def forward(self, z, ls, z_mask):
        """
        Adjust the number of vectors in `z` according to `ls`.
        Return the new `z` and its mask.
        Args:
            z - latent variables, shape: B x L_x x hidden
            ls - target lengths, shape: B
            z_mask - latent mask, shape: B x L_x
        """
        n = z_mask.sum(1)
        arange_l = torch.arange(ls.max().long())
        arange_z = torch.arange(z.size(1))
        if torch.cuda.is_available():
            arange_l = arange_l.cuda()
            arange_z = arange_z.cuda()
        arange_l = arange_l[None, :].repeat(z.size(0), 1).float()
        mu = arange_l * n[:, None].float() / ls[:, None].float()
        arange_z = arange_z[None, None, :].repeat(z.size(0), ls.max().long(), 1).float()
        if OPTS.fp16:
            arange_l = arange_l.half()
            mu = mu.half()
            arange_z = arange_z.half()
        if OPTS.fixbug1:
            logits = - torch.pow(arange_z - mu[:, :, None], 2) / (2. * self.sigma ** 2)
        else:
            distance = torch.clamp(arange_z - mu[:, :, None], -100, 100)
            logits = - torch.pow(2, distance) / (2. * self.sigma ** 2)
        logits = logits * z_mask[:, None, :] - 99. * (1 - z_mask[:, None, :])
        weight = torch.softmax(logits, 2)
        z_prime = (z[:, None, :, :] * weight[:, :, :, None]).sum(2)
        if OPTS.fp16:
            z_prime_mask = (arange_l < ls[:, None].half()).half()
        else:
            z_prime_mask = (arange_l < ls[:, None].float()).float()
        z_prime = z_prime * z_prime_mask[:, :, None]
        return z_prime, z_prime_mask


