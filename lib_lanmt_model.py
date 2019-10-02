#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nmtlab.models.transformer import Transformer
from nmtlab.modules.transformer_modules import TransformerEmbedding
from nmtlab.modules.transformer_modules import PositionalEmbedding
from nmtlab.modules.transformer_modules import LabelSmoothingKLDivLoss
from nmtlab.utils import OPTS
from nmtlab.utils import TensorMap

from lib_lanmt_modules import TransformerCrossEncoder, TransformerEncoder
from lib_lanmt_modules import LengthConverter
from lib_padding import pad_z, pad_z_with_delta
from lib_vae import VAEBottleneck


class LANMTModel(Transformer):

    def __init__(self,
                 enc_layers=3, dec_layers=3,
                 q_layers=6,
                 latent_dim=8,
                 KL_budget=1., KL_weight=1.,
                 budget_annealing=True,
                 **kwargs):
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.q_layers = q_layers
        self.latent_dim = latent_dim
        self.KL_budget = KL_budget
        self.KL_weight = KL_weight
        self.budget_annealing = budget_annealing
        self.training_criteria = "loss"
        super(LANMTModel, self).__init__(**kwargs)

    def prepare(self):
        # Shared embedding layer
        self.x_embed_layer = TransformerEmbedding(self._src_vocab_size, self.embed_size)
        self.y_embed_layer = TransformerEmbedding(self._tgt_vocab_size, self.embed_size)
        self.pos_embed_layer = PositionalEmbedding(self.hidden_size)
        # Length Transformation
        self.length_converter = LengthConverter()
        self.length_embed_layer = nn.Embedding(500, self.hidden_size)
        # encoder and decoder
        self.postz_nn = nn.Linear(self.latent_dim, self.hidden_size)
        self.xz_encoders = nn.ModuleList()
        self.xz_softmax = nn.ModuleList()
        encoder = TransformerEncoder(self.x_embed_layer, self.hidden_size, self.enc_layers)
        self.xz_encoders.append(encoder)
        xz_predictor = nn.Linear(self.hidden_size, self.latent_dim * 2)
        self.xz_softmax.append(xz_predictor)
        self.y_encoder = TransformerEncoder(self.y_embed_layer, self.hidden_size, self.q_layers)
        self.yz_encoder = TransformerCrossEncoder(None, self.hidden_size, self.q_layers)
        self.y_decoder = TransformerCrossEncoder(None, self.hidden_size, self.dec_layers, skip_connect=True)
        # Discretization
        self.bottleneck = VAEBottleneck(self.hidden_size, z_size=self.latent_dim)
        # Length prediction
        self.length_dense = nn.Linear(self.hidden_size, 100)
        # Expander
        self.expander_nn = nn.Linear(self.hidden_size, self._tgt_vocab_size)
        self.label_smooth = LabelSmoothingKLDivLoss(0.1, self._tgt_vocab_size, 0)

        self.set_stepwise_training(False)

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_y(self, x_states, x_mask, y, y_mask):
        y_states = self.y_encoder(y, y_mask)
        if y.size(0) > x_states.size(0) and x_states.size(0) == 1:
            x_states = x_states.expand(y.size(0), -1, -1)
            x_mask = x_mask.expand(y.size(0), -1)
        states = self.yz_encoder(x_states, x_mask, y_states, y_mask)
        return states

    def sample_Q(self, states, sampling=True, prior=None):
        """Return z and p(z|y,x)
        """
        extra = {}
        quantized_vector, code_prob = self.bottleneck(states, sampling=sampling)
        quantized_vector = self.postz_nn(quantized_vector)
        return quantized_vector, code_prob, extra

    def compute_length_pred_loss(self, xz_states, z, z_mask, y_mask):
        y_lens = y_mask.sum(1) - 1
        delta = (y_lens - z_mask.sum(1) + 50.).long().clamp(0, 99)
        mean_z = ((z + xz_states) * z_mask[:, :, None]).sum(1) / z_mask.sum(1)[:, None]
        logits = self.length_dense(mean_z)
        length_loss = F.cross_entropy(logits, delta, reduction="mean")
        length_acc = ((logits.argmax(-1) == delta).float()).mean()
        length_monitors = {
            "lenloss": length_loss,
            "lenacc": length_acc
        }
        return length_monitors

    def compute_vae_KL(self, xz_prob, yz_prob):
        mu1 = yz_prob[:, :, :self.latent_dim]
        var1 = F.softplus(yz_prob[:, :, self.latent_dim:])
        mu2 = xz_prob[:, :, :self.latent_dim]
        var2 = F.softplus(xz_prob[:, :, self.latent_dim:])
        kl = torch.log(var2 / (var1 + 1e-8) + 1e-8) + (
                    (torch.pow(var1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(var2, 2))) - 0.5
        kl = kl.sum(-1)
        return kl

    def gaussian_prob(self, vector, mean, var):
        ret = 1. / torch.sqrt(2 * math.pi * var * var + 1e-8)
        ret *= torch.exp(- (vector - mean) ** 2 / (2 * var * var + 1e-8))
        return ret

    def compute_final_loss(self, yz_prob, xz_prob, x_mask, score_map):
        """ Register KL divergense and bottleneck loss.
        """
        kl = self.compute_vae_KL(xz_prob, yz_prob)
        # Apply budgets for KL divergence: KL = max(KL, budget)
        budget_upperbound = self.KL_budget
        if self.budget_annealing:
            step = OPTS.trainer.global_step()
            half_maxsteps = float(OPTS.maxsteps / 2)
            if step > half_maxsteps:
                rate = (float(step) - half_maxsteps) / half_maxsteps
                min_budget = 0.1
                budget = min_budget + (budget_upperbound - min_budget) * (1. - rate)
        else:
            budget = self.KL_budget
        score_map["KL_budget"] = torch.tensor(budget)
        # Compute KL divergence
        max_mask = ((kl - budget) > 0.).float()
        kl = kl * max_mask + (1. - max_mask) * budget
        kl_loss = (kl * x_mask).sum() / x_mask.shape[0]
        # Report KL divergence
        score_map["kl"] = kl_loss
        # Also report the averge KL for each token
        score_map["tok_kl"] = (kl * x_mask).sum() / x_mask.sum()
        # Report cross-entropy loss
        score_map["nll"] = score_map["loss"]
        # Cross-entropy loss is *already* backproped when computing softmaxes in shards
        # So only need to compute the remaining losses and then backprop them
        remain_loss = score_map["kl"].clone() * self.KL_weight
        if "lenloss" in score_map:
            remain_loss += score_map["lenloss"]
        # Report the combined loss
        score_map["loss"] = remain_loss + score_map["nll"]
        return score_map, remain_loss

    def forward(self, x, y, sampling=False, return_code=False):
        """Forward to compute the loss.
        """
        score_map = {}
        x_mask = torch.ne(x, 0).float()
        y_mask = torch.ne(y, 0).float()
        # Compute p(z|x)
        xz_states = self.xz_encoders[0](x, x_mask)
        full_xz_states = xz_states
        xz_prob = self.xz_softmax[0](xz_states)
        # Compute p(z|y,x) and sample z
        yz_states = self.encode_y(self.x_embed_layer(x), x_mask, y, y_mask)
        # Create latents
        z_mask = x_mask
        z, yz_prob, _ = self.sample_Q(yz_states)
        # Comute length loss
        length_scores = self.compute_length_pred_loss(xz_states, z, z_mask, y_mask)
        score_map.update(length_scores)
        z_expand, z_expand_mask = z, z_mask
        tgt_states_mask = y_mask
        # Padding z to fit target states
        z_pad, _ = pad_z(self, z_expand, z_expand_mask, tgt_states_mask)

        # --------------------------  Decoder -------------------------
        decoder_states = self.y_decoder(z_pad, y_mask, full_xz_states, x_mask)
        # Compute loss
        decoder_outputs = TensorMap({"final_states": decoder_states})
        denom = x.shape[0]
        if self._shard_size is not None and self._shard_size > 0:
            loss_scores, decoder_tensors, decoder_grads = self.compute_shard_loss(
                decoder_outputs, y, y_mask, denominator=denom, ignore_first_token=False, backward=False
            )
            loss_scores["word_acc"] *= float(y_mask.shape[0]) / y_mask.sum().float()
            score_map.update(loss_scores)
        else:
            raise SystemError("Shard size must be setted or the memory is not enough for this model.")

        score_map, remain_loss = self.compute_final_loss(yz_prob, xz_prob, z_mask, score_map)
        # Backward for shard loss
        if self._shard_size is not None and self._shard_size > 0 and decoder_tensors is not None:
            decoder_tensors.append(remain_loss)
            decoder_grads.append(None)
            torch.autograd.backward(decoder_tensors, decoder_grads)
        return score_map

    def sample_z(self, z_prob):
        """ Return the quantized vector given probabiliy distribution over z.
        """
        quantized_vector = z_prob[:, :, :self.latent_dim]
        quantized_vector = self.postz_nn(quantized_vector)
        return quantized_vector

    def predict_length(self, xz_states, z, z_mask, refinement=False):
        mean_z = ((z + xz_states) * z_mask[:, :, None]).sum(1) / z_mask.sum(1)[:, None]
        logits = self.length_dense(mean_z)
        if OPTS.Tsearchlen and not refinement:
            deltas = torch.argsort(logits, 1, descending=True)
            n_samples = 3 if OPTS.Tsearchz else 9
            delta = deltas[:, :n_samples] - 50
        else:
            delta = logits.argmax(-1) - 50
        return delta

    def translate(self, x, y=None, q=None, xz_states=None):
        """ Testing code
        """
        if OPTS.Tsearchz or OPTS.Tsearchlen:
            return self.beam_translate(x, y=y, q=q, xz_states=xz_states)
        x_mask = torch.ne(x, 0).float()
        # Compute p(z|x)
        if xz_states is None:
            xz_states = self.xz_encoders[0](x, x_mask)
        # Sample a z
        if q is not None:
            # Z is provided
            z = q
        elif y is not None:
            # Y is provided
            y_mask = torch.ne(y, 0).float()
            x_embeds = self.x_embed_layer(x)
            yz_states = self.encode_y(x_embeds, x_mask, y, y_mask)
            _, yz_prob, bottleneck_scores = self.sample_Q(yz_states)
            z = self.sample_z(yz_prob)
        else:
            # Compute prior to get Z
            xz_prob = self.xz_softmax[0](xz_states)
            if OPTS.mixprior > 0:
                weights = xz_prob[:, :, :OPTS.mixprior]
                pidx = weights.argmax(2)
                assert pidx.shape[0] == 1
                selector = torch.arange(self.latent_dim * 2)[None, :].repeat(pidx.shape[1], 1)
                if torch.cuda.is_available():
                    selector = selector.cuda()
                selector = selector + (pidx[0][:, None] * self.latent_dim * 2)
                xz_prob = xz_prob[:, torch.arange(xz_prob.shape[0]), selector]
            z = self.sample_z(xz_prob)
        # Predict length
        if y is None or True:
            length_delta = self.predict_length(xz_states, z, x_mask)
        else:
            length_delta = (y_mask.sum(1) - 1 - x_mask.sum(1)).long()
        # Padding z to cover the length of y
        z_pad, z_pad_mask, y_lens = pad_z_with_delta(self, z, x_mask, length_delta + 1)
        if z_pad.size(1) == 0:
            return None, y_lens, xz_prob.argmax(-1)
        # Run decoder to predict the target words
        decoder_states = self.y_decoder(z_pad, z_pad_mask, xz_states, x_mask)
        # Get the predictions
        logits = self.expander_nn(decoder_states)
        pred = logits.argmax(-1)

        return pred, y_lens, z, xz_states

    def beam_translate(self, x, y=None, q=None, xz_states=None, max_candidates=None):
        if max_candidates is None:
            max_candidates = OPTS.Tncand
        x_mask = torch.ne(x, 0).float()
        # Compute p(z|x)
        if xz_states is None:
            xz_states = self.xz_encoders[0](x, x_mask)
        # Sample a z
        if q is not None:
            # Z is provided
            z = q
        elif y is not None:
            # Y is provided
            y_mask = torch.ne(y, 0).float()
            x_embeds = self.x_embed_layer(x)
            yz_states = self.encode_y(x_embeds, x_mask, y, y_mask)
            _, yz_prob, bottleneck_scores = self.sample_Q(yz_states)
            z = self.sample_z(yz_prob)
        else:
            # Compute prior to get Z
            xz_prob = self.xz_softmax[0](xz_states)
            if OPTS.Tsearchz:
                n_samples = int(math.sqrt(max_candidates)) if OPTS.Tsearchlen else max_candidates
                z = self.bottleneck.sample_any_dist(xz_prob, samples=n_samples, noise_level=0.5)
                z = self.postz_nn(z)
            else:
                z = self.sample_z(xz_prob)
        # Predict length
        if y is None:
            length_delta = self.predict_length(xz_states, z, x_mask, refinement=q is not None)
        else:
            length_delta = (y_mask.sum(1) - 1 - x_mask.sum(1)).long()
        # Padding z to cover the length of y
        if OPTS.Tsearchlen and q is None:
            if OPTS.Tsearchz:
                n_samples = z.size(0)
                z = z.unsqueeze(1).expand(-1, n_samples, -1, -1).contiguous().view(-1, z.size(1), z.size(2))
                length_delta = length_delta.flatten()
                x_mask = x_mask.expand(z.size(0), -1)
            elif z.size(0) < length_delta.size(1):
                z = z.expand(length_delta.size(1), -1, -1)
                x_mask = x_mask.expand(length_delta.size(1), -1)
                length_delta = length_delta[0]
        z_pad, z_pad_mask, y_lens = pad_z_with_delta(self, z, x_mask, length_delta + 1)
        assert z_pad.size(1) > 0
        # Run decoder to predict the target words
        decoder_states = self.y_decoder(z_pad, z_pad_mask, xz_states, x_mask)
        # Get the predictions
        logits = self.expander_nn(decoder_states)
        if (OPTS.Tsearchz or OPTS.Tsearchlen) and not OPTS.Trescore:
            if q is not None or OPTS.Tgibbs < 1:
                logprobs, preds = torch.log_softmax(logits, 2).max(2)
                logprobs = (logprobs * z_pad_mask).sum(1)  # x batch x 1
                preds = preds * z_pad_mask.long()
                # after deterimistic refinement
                pred = preds[logprobs.argmax()].unsqueeze(0)
            else:
                pred = logits.argmax(-1)
                pred = pred * z_pad_mask.long()
        else:
            pred = logits.argmax(-1)

        return pred, y_lens, z, xz_states

    def compute_Q(self, x, y):
        """Forward to compute the loss.
        """
        x_mask = torch.ne(x, 0).float()
        y_mask = torch.ne(y, 0).float()
        # Compute p(z|y,x) and sample z
        yz_states = self.encode_y(self.x_embed_layer(x), x_mask, y, y_mask)
        z, yz_prob, _ = self.sample_Q(yz_states, sampling=False)
        return z, yz_prob

    def load_state_dict(self, state_dict):
        """Remove deep generative model weights.
        """
        super(LANMTModel, self).load_state_dict(state_dict, strict=True)

    def measure_ELBO(self, x, y):
        """Measure the ELBO in the inference time."""
        x_mask = torch.ne(x, 0).float()
        y_mask = torch.ne(y, 0).float()
        # Compute p(z|x)
        xz_states = self.xz_encoders[0](x, x_mask)
        xz_prob = self.xz_softmax[0](xz_states)
        # Compute p(z|y,x) and sample z
        yz_states = self.encode_y(self.x_embed_layer(x), x_mask, y, y_mask)
        # Sampling for 20 times
        likelihood_list = []
        for _ in range(20):
            z, yz_prob, bottleneck_scores = self.sample_Q(yz_states)
            z_pad, _ = pad_z(self, z, x_mask, y_mask)
            decoder_states = self.y_decoder(z_pad, y_mask, xz_states, x_mask)
            logits = self.expander_nn(decoder_states)
            likelihood = - F.cross_entropy(logits[0], y[0], reduction="sum")
            likelihood_list.append(likelihood)
        kl = self.compute_vae_KL(xz_prob, yz_prob).sum()
        mean_likelihood = sum(likelihood_list) / len(likelihood_list)
        elbo = mean_likelihood - kl
        return elbo

