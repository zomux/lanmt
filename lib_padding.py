#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

from nmtlab.utils import OPTS


def pad_xy_length(x, y=None):
    if OPTS.conv > 0:
        # pad x and y to be a multiple of 2^OPTS.conv
        base = 2 ** OPTS.conv
        x_len = x.shape[1]
        adjusted_len = math.ceil(x_len / float(base)) * base
        padding = x.new_zeros(x.shape[0], adjusted_len - x_len)
        x = torch.cat([x, padding], dim=1)
        # if y is not None:
        #     y_len = y.shape[1]
        #     adjusted_len = math.ceil(y_len / float(base)) * base
        #     padding = y.new_zeros(y.shape[0], adjusted_len - y_len)
        #     y = torch.cat([y, padding], dim=1)
    return x, y


def pad_z(model, z, z_mask, y_mask):
    """
            If len(z) > len(y):
                do nothing, and put padding positions into y_mask
            If len(z) < len(y):
                pad Z with paddings
            Finally, add positional embeddngs
            Return: padded z, and the mask for Zs and PADs
            the mask shall be also used on y for computing the reconstruction loss
            """
    if OPTS.lenconv:
        rc = 1. / math.sqrt(2)
        lens = y_mask.sum(-1)
        converted_states, _ = model.length_converter(z, lens, z_mask)
        pos_embed = model.pos_embed_layer(converted_states)
        len_embed = model.length_embed_layer(lens.long())
        converted_states = rc * converted_states + 0.5 * pos_embed + 0.5 * len_embed[:, None, :]
        return converted_states, y_mask
    else:
        z = z.clone()
        z_lens = z_mask.sum(1).long()
        # y_mask = y_mask[:, 1:]
        # y_lens = y_mask.sum(1).long()
        # Create pos embeds
        max_z_len = z_mask.shape[1]
        max_y_len = y_mask.shape[1]
        if max_y_len > max_z_len:
            delta = max_y_len - max_z_len
            padding = z.new_zeros(z.size(0), delta, z.size(2))
            z = torch.cat([z, padding], 1)
        # Make masks
        max_z_len = max_z_len if max_z_len > max_y_len else max_y_len
        # length of z >= length of y is garenteed
        pos_matrix = torch.arange(max_z_len).long()[None, :].repeat(z_lens.shape[0], 1).cuda()
        extended_z_mask = (pos_matrix < z_lens[:, None]).float()  # <- this the z mask including the PAD
        # extended_y_mask = (pos_matrix < y_lens[:, None]).float()
        # Change padded positions into padding embedding
        pad_embed = model.x_embed_layer.weight[3]
        padding_embed_matrix = pad_embed.repeat(z.size(0), z.size(1), 1)
        z = z * extended_z_mask[:, :, None] + padding_embed_matrix * (1 - padding_embed_matrix)
        # Make extended mask with padded positions
        # z_pad_mask = extended_z_mask.clone()
        # z_pad_mask[:, :max_y_len] = ((z_pad_mask[:, :max_y_len] + y_mask) > 0.).float()
        z = z[:, :max_y_len]
        # Add postional embedding
        z += model.pos_embed_layer(z)
        return z, y_mask


def pad_z_with_delta(model, z, z_mask, delta):
        """
        Pad z with delta
        If len(z) > |z| + delta:
            do nothing, and put padding positions into y_mask
        If len(z) < |z| + delta:
            pad Z with paddings
        """
        z = z.clone()
        z_lens = z_mask.sum(1).long()
        y_lens = z_lens + delta
        if OPTS.lenconv:
            rc = 1. / math.sqrt(2)
            lens = y_lens
            converted_states, _ = model.length_converter(z, lens, z_mask)
            pos_embed = model.pos_embed_layer(converted_states)
            len_embed = model.length_embed_layer(lens.long())
            converted_states = rc * converted_states + 0.5 * pos_embed + 0.5 * len_embed[:, None, :]
            arange = torch.arange(lens.max().long())
            if torch.cuda.is_available():
                arange = arange.cuda()
            y_mask = (arange[None, :].repeat(z.size(0), 1) < lens[:, None]).float()
            return converted_states, y_mask, y_lens
        else:
            # Create pos embeds
            max_z_len = z_mask.shape[1]
            max_y_len = y_lens.max().long()
            if max_y_len > max_z_len:
                delta = max_y_len - max_z_len
                padding = z.new_zeros(z.size(0), delta, z.size(2))
                z = torch.cat([z, padding], 1)
            # Make masks
            max_z_len = max_z_len if max_z_len > max_y_len else max_y_len
            # length of z >= length of y is garenteed
            pos_matrix = torch.arange(max_z_len).long()[None, :].repeat(z_lens.shape[0], 1)
            if torch.cuda.is_available():
                pos_matrix = pos_matrix.cuda()
            extended_z_mask = (pos_matrix < z_lens[:, None]).float()  # <- this the z mask including the PAD
            y_mask = (pos_matrix < y_lens[:, None]).float()
            # Change padded positions into padding embedding
            pad_embed = model.x_embed_layer.weight[3]
            padding_embed_matrix = pad_embed.repeat(z.size(0), z.size(1), 1)
            z = z * extended_z_mask[:, :, None] + padding_embed_matrix * (1 - padding_embed_matrix)
            # Make extended mask with padded positions
            z_pad_mask = y_mask[:, :max_y_len]
            z = z[:, :max_y_len]
            # Add postional embedding
            z += model.pos_embed_layer(z)
            return z, z_pad_mask, y_lens