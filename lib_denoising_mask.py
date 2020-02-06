#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenoiseMask(nn.Module):
    """
    Produce (1, size, size) mask for masking out previous positions.
    """

    def __init__(self, max_len=1000):
        super(DenoiseMask, self).__init__()
        shape = (1, max_len, max_len)
        subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
        mask = (torch.from_numpy(subsequent_mask) == 0).float()
        mask = 1. - mask * mask.transpose(1, 2)
        self.register_buffer("mask", mask)

    def forward(self, x):
        """Compute the temporal mask on given embeddings

        Args:
            x - embedding ~ (batch, len, size)
        """
        if type(x) == int:
            seq_len = x
        else:
            seq_len = x.shape[-2]
        return self.mask[:, :seq_len, :seq_len]
