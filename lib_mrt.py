#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lanmt.lib_lanmt_modules import TransformerEncoder
from nmtlab.models import Transformer


class MinRiskNetwork(Transformer):

    def __init__(self, lanmt_model, hidden_size=512, latent_size=8):
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        super(MinRiskNetwork, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]

    def prepare(self):
        self._encoder = TransformerEncoder(None, self._hidden_size, 3)
        self._hidden2latent = nn.Linear(self._hidden_size, self._latent_size)

    def compute_latent_prob(self, embeds, x_mask):
        states = self.encoder(embeds, mask=x_mask)
        z_mean = self._hidden2latent(states)
        return z_mean

    def compute_loss(self, x, y):
        x_mask = torch.ne(x, 0).float()
        y_mask = torch.ne(y, 0).float()
        with torch.no_grad():
            embeds = self.nmt().x_embed_layer(x)
        z_mean = self.compute_latent_prob(embeds, x_mask)
        
        sigma = 1.
        noised_oracle = oracle_latent + torch.randn_like(oracle_latent) * sigma
        target = oracle_latent
        pred = self.denoise(noised_oracle, x_states, mask)
        loss = 0.5 * (pred - target).pow(2).sum(2)
        loss = (loss * mask).sum(1) / mask.sum(1)
        loss = loss.mean() * 100.
        return {"loss": loss, "abs": abs(pred - target).mean(), "target_abs": abs(target).mean()}

    def forward(self, x, y, sampling=False):
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))
        with torch.no_grad():
            x_states = self._lanmt[0].x_embed_layer(x)
            q_states = self._lanmt[0].compute_Q_states(x_states, x_mask, y, y_mask)
            oracle_z, _ = self._lanmt[0].bottleneck(q_states, sampling=False)
        score_map = self.compute_loss(oracle_z.detach(), x_states.detach(), x_mask)
        return score_map

    def refine(self, z, x_states, mask=None, n_steps=10, step_size=0.001):
        n_steps = 10
        with torch.no_grad():
            for _ in range(n_steps):
                # final_z = z * 0.
                # for _ in range(5):
                #     final_z += self.denoise(z + torch.randn_like(z), x_states, mask) / 5
                # z = final_z
                new_z = self.denoise(z, x_states, mask)
                norm = (new_z - z).norm(dim=2)
                index = norm.argmax(1)
                z[:, index] = new_z[:, index]
        return z

    def nmt(self):
        return self._lanmt[0]

