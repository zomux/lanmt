#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import sys
from nmtlab.utils import OPTS
from collections import defaultdict

def analyze_latents(nmt, src_vocab, tgt_vocab, src_corpus, tgt_corpus):
    src_lines = open(src_corpus).readlines()
    tgt_lines = open(tgt_corpus).readlines()
    stats = defaultdict(list)
    for src_line, tgt_line in zip(src_lines[:100], tgt_lines[:100]):
        src_tokens = "<s> {} </s>".format(src_line.strip()).split()
        tgt_tokens = "<s> {} </s>".format(tgt_line.strip()).split()
        src_tokens = src_vocab.encode(src_tokens)
        tgt_tokens = tgt_vocab.encode(tgt_tokens)
        x = torch.tensor([src_tokens])
        y = torch.tensor([tgt_tokens])
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            x_mask = torch.ne(x, 0).float()
            y_mask = torch.ne(y, 0).float()
            prior_states = nmt.prior_encoder(x, x_mask)
            if OPTS.zeroprior:
                prior_prob = nmt.standard_gaussian_dist(1, x.shape[1])
            else:
                prior_prob = nmt.prior_prob_estimator(prior_states)
            # prior_mean = prior_prob[:, :, :nmt.latent_dim]
            prior_mean = nmt.bottleneck.sample_any_dist(prior_prob, noise_level=0.5)
            q_states = nmt.compute_Q_states(nmt.x_embed_layer(x), x_mask, y, y_mask)
            oracle_z, _ = nmt.bottleneck(q_states, sampling=False)
            stats["abs"].append(abs(oracle_z - prior_mean).mean().cpu().numpy())
            stats["abs_var"].append(abs(oracle_z - prior_mean).var().cpu().numpy())
            stats["abs_max"].append(abs(oracle_z - prior_mean).max().cpu().numpy())
            if OPTS.scorenet:
                refined_z = OPTS.scorenet.refine(prior_mean, nmt.x_embed_layer(x), x_mask, n_steps=50, step_size=1.)
                stats["refined_abs"].append(abs(oracle_z - refined_z).mean().cpu().numpy())
                # if stats["refined_abs"][-1] > stats["abs"][-1]:
                #     import pdb;pdb.set_trace()

    for key, vals in stats.items():
        print(key, "=", np.mean(vals))

