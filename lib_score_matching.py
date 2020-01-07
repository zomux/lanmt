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


class LatentScoreNetwork(Transformer):

    def __init__(self, lanmt_model, hidden_size=512, latent_size=8):
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self.set_stepwise_training(False)
        super(LatentScoreNetwork, self).__init__(src_vocab_size=1, tgt_vocab_size=1)
        lanmt_model.train(False)
        self._lanmt = [lanmt_model]

    def prepare(self):
        self._encoder = TransformerEncoder(None, self._hidden_size, 3)
        self._latent2hidden = nn.Linear(self._latent_size, self._hidden_size)
        self._hidden2latent = nn.Linear(self._hidden_size, self._latent_size)

    def scorenet(self, latent, x_states, mask):
        h = self._latent2hidden(latent)
        h = self._encoder(torch.cat([h, x_states], 1), mask=mask)
        scores = self._hidden2latent(h[:, :latent.shape[1]])
        return scores

    def compute_loss(self, oracle_latent, x_states, mask):
        sigma = 1.
        noised_oracle = oracle_latent + torch.randn_like(oracle_latent) * sigma
        target = - (noised_oracle - oracle_latent) / (sigma ** 2)
        scores = self.scorenet(noised_oracle, x_states, torch.cat([mask, mask], 1))
        loss = 0.5 * (scores - target).pow(2).sum(2)
        loss = (loss * mask).sum(1) / mask.sum(1)
        loss = loss.mean() * 100.
        return {"loss": loss, "abs": abs(scores - target).mean(), "target_abs": abs(target).mean()}

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
        if mask is not None:
            mask = torch.cat([mask, mask], 1)
        with torch.no_grad():
            for _ in range(n_steps):
                grad = self.scorenet(z, x_states, mask)
                noise = torch.randn_like(z) * np.sqrt(step_size * 2)
                z = z + step_size * grad + noise
        return z

    def nmt(self):
        return self._lanmt[0]

