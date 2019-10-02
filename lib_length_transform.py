#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLengthPredModule(nn.Module):

    def __init__(self):
        super(DeepLengthPredModule, self).__init__()
        self.dense = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 100))

    def forward(self, latents, x_states, x_mask):
        x = latents + x_states
        logits = self.dense(x)
        logits = (logits * x_mask[:, :, None]).sum(1) / x_mask.sum(1)[:, None]
        return logits
