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
from nmtlab.utils import MapDict, TensorMap
from mcgen.lib_namt_modules import TransformerCrossEncoder, TransformerEncoder
from mcgen.lib_namt_modules import LengthConverter
from mcgen.lib_rupdate_model import LatentUpdateModel
from mcgen.lib_latent_rerank import LatentRerankModel
from mcgen.lib_padding import pad_z, pad_z_with_delta
from mcgen.lib_textglow import get_base_logprob

# !!! Deprecated

class LANMTModel(Transformer):

    def __init__(self, enc_layers=3, dec_layers=3, **kwargs):
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.approx_layers = 6
        # self.training_criteria = "tok+kl"
        self.training_criteria = "loss"
        assert OPTS.bottleneck == "vae"
        if OPTS.rupdate or OPTS.zrerank or OPTS.priorft:
            self.training_criteria = "loss"
        super(LANMTModel, self).__init__(**kwargs)

    def prepare(self):
        # Shared embedding layer
        max_size = self._src_vocab_size if self._src_vocab_size > self._tgt_vocab_size else self._tgt_vocab_size
        self.x_embed_layer = TransformerEmbedding(max_size, self.embed_size)
        self.y_embed_layer = TransformerEmbedding(self._tgt_vocab_size, self.embed_size)
        self.pos_embed_layer = PositionalEmbedding(self.hidden_size)
        # Length Transform
        self.length_converter = LengthConverter()
        self.length_embed_layer = nn.Embedding(500, self.hidden_size)
        # encoder and decoder
        if OPTS.zdim > 0:
            z_size = OPTS.zdim
            self.postz_nn = nn.Linear(z_size, self.hidden_size)
        else:
            z_size = self.hidden_size
        self.latent_size = z_size
        self.xz_encoders = nn.ModuleList()
        self.xz_softmax = nn.ModuleList()
        encoder = TransformerEncoder(self.x_embed_layer, self.hidden_size, self.enc_layers)
        self.xz_encoders.append(encoder)
        if OPTS.mixprior > 0:
            out_size = z_size * 2 * OPTS.mixprior + OPTS.mixprior
            xz_predictor = nn.Linear(self.hidden_size, out_size)
        else:
            xz_predictor = nn.Linear(self.hidden_size, z_size * 2)
        self.xz_softmax.append(xz_predictor)
        self.y_encoder = TransformerEncoder(self.y_embed_layer, self.hidden_size, self.approx_layers)
        self.yz_encoder = TransformerCrossEncoder(None, self.hidden_size, self.approx_layers)
        self.y_decoder = TransformerCrossEncoder(None, self.hidden_size, self.dec_layers, skip_connect=True)
        # Discretization
        if OPTS.bottleneck == "vae":
            from mcgen.lib_vae import VAEBottleneck
            self.bottleneck = VAEBottleneck(self.hidden_size, z_size=z_size)
        else:
            raise NotImplementedError
        # Length prediction
        self.length_dense = nn.Linear(self.hidden_size, 100)
        # Expander
        self.expander_nn = nn.Linear(self.hidden_size, self._tgt_vocab_size)
        self.label_smooth = LabelSmoothingKLDivLoss(0.1, self._tgt_vocab_size, 0)
        if OPTS.flowft:
            from mcgen.lib_textglow import TextGlow
            self.flow = TextGlow()
        # Latent update model
        if OPTS.rupdate:
            for param in self.parameters():
                param.requires_grad = False
            self.rupdate_nn = LatentUpdateModel(self.hidden_size, z_size)
        if OPTS.zrerank:
            for param in self.parameters():
                param.requires_grad = False
            self.rerank_nn = LatentRerankModel(self.hidden_size, z_size)
        # Progressive training
        if OPTS.phase == "limitx_2" or OPTS.phase == "limity_2":
            self.y_encoder2 = TransformerEncoder(self.y_embed_layer, self.hidden_size, self.enc_layers)
            self.yz_encoder2 = TransformerCrossEncoder(None, self.hidden_size, self.enc_layers)
            self.x_encoder2 = TransformerEncoder(None, self.hidden_size, self.enc_layers)
            self.xz_predictor2 = nn.Linear(self.hidden_size, z_size * 2)
            self.x_fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.bottleneck2 = VAEBottleneck(self.hidden_size, z_size=z_size)
            self.postz_nn2 = nn.Linear(z_size, self.hidden_size)
            fixed_layers = [
                self.x_embed_layer, self.y_embed_layer, self.pos_embed_layer, self.xz_encoders, self.xz_softmax,
                self.yz_encoder, self.y_encoder,
                self.bottleneck, self.postz_nn
            ]
            for layer in fixed_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        # Prior finetune
        if OPTS.priorft or OPTS.flowft:
            fixed_layers = [
                self.x_embed_layer, self.y_embed_layer, self.pos_embed_layer,
                self.y_decoder, self.expander_nn, self.length_dense,
                self.length_converter, self.xz_encoders,
                self.yz_encoder, self.y_encoder,
                self.bottleneck, self.postz_nn
            ]
            for layer in fixed_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.set_stepwise_training(False)

    def initialize_parameters(self):
        if OPTS.flowft:
            return
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

    def compute_q2_states(self, x_states, x_mask, y, y_mask):
        y_states = self.y_encoder2(y, y_mask)
        states = self.yz_encoder2(x_states, x_mask, y_states, y_mask)
        return states

    def sample_Q(self, states, sampling=True, prior=None):
        """Return z and p(z|y,x)
        """
        extra = {}
        if OPTS.bottleneck == "vae":
            quantized_vector, code_prob = self.bottleneck(states, sampling=sampling)
            quantized_vector = self.postz_nn(quantized_vector)
        else:
            raise NotImplementedError
        return quantized_vector, code_prob, extra

    def sample_Q2(self, states, sampling=True):
        sampled_vector, prob = self.bottleneck(states, sampling=sampling)
        sampled_vector = self.postz_nn(sampled_vector)
        return sampled_vector, prob, None

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
        if OPTS.mixprior > 0:
            return self.compute_mixprior_KL(xz_prob, yz_prob)
        mu1 = yz_prob[:, :, :self.latent_size]
        var1 = F.softplus(yz_prob[:, :, self.latent_size:])
        mu2 = xz_prob[:, :, :self.latent_size]
        var2 = F.softplus(xz_prob[:, :, self.latent_size:])
        kl = torch.log(var2 / (var1 + 1e-8) + 1e-8) + (
                    (torch.pow(var1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(var2, 2))) - 0.5
        kl = kl.sum(-1)
        return kl

    def gaussian_prob(self, vector, mean, var):
        ret = 1. / torch.sqrt(2 * math.pi * var * var + 1e-8)
        ret *= torch.exp(- (vector - mean) ** 2 / (2 * var * var + 1e-8))
        return ret

    def compute_flow_KL(self, x_states, q_prob, xz_prob=None, z_mask=None):
        n_samples = 3
        zs = self.bottleneck.sample_any_dist(q_prob, samples=n_samples)
        zs = zs.detach()
        mu_q = q_prob[:, :, :self.latent_size]
        var_q = F.softplus(q_prob[:, :, self.latent_size:])
        x_states = x_states.permute(0, 2, 1)
        total_kl = 0.
        mean_baseprob = 0.
        # deterministic version
        # base_z, log_s_list, log_det_W_list = self.flow(x_states, mu_q.permute(0, 2, 1), z_mask)
        # base_logprob = get_base_logprob(base_z, log_s_list, log_det_W_list, z_mask)
        # total_kl = - base_logprob / z_mask.sum()
        # mean_baseprob = (self.gaussian_prob(base_z, base_z * 0, base_z*0+1.0) * z_mask[:, None, :]).sum() / z_mask.sum()
        # mean_baseprob = (torch.abs(base_z) * z_mask[:, None, :]).sum() / z_mask.sum() / self.latent_size
        for k in range(n_samples):
            z = zs[:, k]
            log_qprob = - torch.log(torch.sqrt(2 * math.pi * var_q * var_q)) - (z - mu_q)*(z - mu_q) / (2*var_q*var_q)
            log_qprob = (log_qprob * z_mask[:, :, None]).sum()
            base_z, log_s_list, log_det_W_list = self.flow(x_states, z.permute(0, 2, 1), z_mask)
            base_logprob = get_base_logprob(base_z, log_s_list, log_det_W_list, z_mask)
            total_kl += log_qprob - base_logprob
            # mean_baseprob += self.gaussian_prob(base_z, base_z * 0, base_z*0+1.0).mean()
            mean_baseprob += (torch.abs(base_z) * z_mask[:, None, :]).sum() / z_mask.sum() / self.latent_size
        total_kl /= x_states.size(0) * n_samples
        mean_baseprob /= n_samples
        return total_kl, mean_baseprob


    def compute_mixprior_KL(self, xz_prob, yz_prob):
        z_from_y = self.bottleneck.sample_any_dist(yz_prob)
        z_from_y = z_from_y.detach()
        mu_y = yz_prob[:, :, :self.latent_size]
        var_y = F.softplus(yz_prob[:, :, self.latent_size:])
        kl_nom = torch.log(self.gaussian_prob(z_from_y, mu_y, var_y) + 1e-8)
        weights = xz_prob[:, :, :OPTS.mixprior]
        weights = F.softmax(weights, dim=2)
        priors = xz_prob[:, :, OPTS.mixprior:]
        total_prior = None
        for k in range(OPTS.mixprior):
            mean = priors[:, :, k * self.latent_size * 2: k * self.latent_size * 2 + self.latent_size]
            var = F.softplus(priors[:, :, k * self.latent_size * 2 + self.latent_size: (k + 1) * self.latent_size * 2])
            sub_prob = weights[:, :, k][:, :, None] * self.gaussian_prob(z_from_y, mean, var)
            if total_prior is None:
                total_prior = sub_prob
            else:
                total_prior += sub_prob
        kl_denom = torch.log(total_prior + 1e-8)
        kl = (kl_nom - kl_denom).sum(-1)
        return kl

    def compute_final_loss(self, yz_prob, xz_prob, x_mask, score_map):
        """ Register KL divergense and bottleneck loss.
        """
        if not OPTS.withkl:
            yz_prob = yz_prob.detach()
        if OPTS.bottleneck == "vae":
            kl = self.compute_vae_KL(xz_prob, yz_prob)
        else:
            raise NotImplementedError
        if OPTS.klbudget:
            budget = float(OPTS.budgetn) / 100.
            if OPTS.annealkl and not OPTS.klft and not OPTS.origft:
                step = OPTS.trainer.global_step()
                half_maxsteps = float(OPTS.maxsteps / 2)
                if step > half_maxsteps:
                    rate = (float(step) - half_maxsteps) / half_maxsteps
                    min_budget = 0.1
                    budget = min_budget + (budget - min_budget) * (1. - rate)
                score_map["budget"] = torch.tensor(budget)
            max_mask = ((kl - budget) > 0.).float()
            kl = kl * max_mask + (1. - max_mask) * budget
        if OPTS.sumloss:
            kl_loss = (kl * x_mask).sum() / x_mask.shape[0]
            score_map["wkl"] = (kl * x_mask).sum() / x_mask.sum()
        else:
            kl_loss = (kl * x_mask).sum() / x_mask.sum()
        score_map["kl"] = kl_loss
        # Combine all losses
        score_map["tokloss"] = score_map["loss"]
        score_map["tok+kl"] = score_map["loss"] + kl_loss
        if OPTS.withkl:
            klweight = float(OPTS.klweight) / 100
            shard_loss = score_map["kl"].clone() * klweight
        else:
            shard_loss = score_map["kl"].clone()
        if "neckloss" in score_map:
            shard_loss += score_map["neckloss"]
        if "lenloss" in score_map:
            if OPTS.sumloss:
                shard_loss += score_map["lenloss"] * float(OPTS.lenweight)
            else:
                shard_loss += score_map["lenloss"] * 0.1
        score_map["shard_loss"] = shard_loss
        score_map["loss"] = shard_loss + score_map["tokloss"]
        return score_map

    def forward(self, x, y, sampling=False, return_code=False):
        """Forward to compute the loss.
        """
        score_map = {}
        x_mask = torch.ne(x, 0).float()
        y_mask = torch.ne(y, 0).float()
        # Compute p(z|x)
        if not OPTS.phase.endswith("_2"):
            xz_states = self.xz_encoders[0](x, x_mask)
            full_xz_states = xz_states
            xz_prob = self.xz_softmax[0](xz_states)
        # Compute p(z|y,x) and sample z
        if OPTS.phase == "limitx_1" or OPTS.phase == "limitx_2":
            tile = torch.arange(1, x_mask.shape[1], 2)
            tiled_x_mask = x_mask.clone()
            tiled_x_mask[:, tile] = 0
            yz_states = self.encode_y(self.x_embed_layer(x), tiled_x_mask, y, y_mask)
        else:
            yz_states = self.encode_y(self.x_embed_layer(x), x_mask, y, y_mask)
        # Create latents
        z_mask = x_mask
        z, yz_prob, _ = self.sample_Q(yz_states)
        if OPTS.phase == "limitx_2" or OPTS.phase == "limity_2":
            # In the second phase of progressive training, compute z~q2 and p(z'|x, z)
            z_q1 = z
            fused_x_embeds = self.x_fusion(torch.cat([self.x_embed_layer(x), z_q1], dim=2))
            xz_states = self.x_encoder2(fused_x_embeds, x_mask)
            full_xz_states = xz_states
            xz_prob = self.xz_predictor2(xz_states)
            # Compute z~q2
            yz_states = self.compute_q2_states(xz_states, x_mask, y, y_mask)
            z, yz_prob, _ = self.sample_Q2(yz_states)
        # Latent update model
        if OPTS.rupdate:
            # Train the latent update model with KL(r || q)
            z_from_x = self.bottleneck.sample_any_dist(yz_prob, deterministic=False)
            z_from_x = self.postz_nn(z_from_x).detach()
            r_prob = self.rupdate_nn(z_from_x, x_mask, xz_states.detach(), x_mask)
            # Distance-based loss
            # target_vec = yz_prob[:, :, :self.latent_size].detach()
            # pred_vec = r_prob[:, :, :self.latent_size]
            # loss = 0.5 * (target_vec - pred_vec).pow(2).sum(-1).mean()
            # ret = {"loss": loss}
            kl = self.compute_vae_KL(r_prob, yz_prob.detach())
            loss = (kl * x_mask).sum() / x_mask.shape[0]
            ret = {"loss": loss, "wkl": (kl * x_mask).sum() / x_mask.sum()}
            if self.training:
                loss.backward()
            return ret
        elif OPTS.zrerank:
            z_from_x = self.bottleneck.sample_any_dist(xz_prob)
            z_from_x = self.postz_nn(z_from_x).detach()
            z_from_y = self.bottleneck.sample_any_dist(yz_prob)
            z_from_y = self.postz_nn(z_from_y).detach()
            hinge_loss = self.rerank_nn(z_from_y, z_from_x, x_mask, xz_states.detach(), x_mask)
            if self.training:
                hinge_loss.backward()
            return {"loss": hinge_loss}
        if OPTS.priorft:
            kl = self.compute_vae_KL(xz_prob, yz_prob.detach())
            kl_loss = (kl * x_mask).sum() / x_mask.shape[0]
            kl_loss = kl_loss.mean()
            score_map["wkl"] = (kl * x_mask).sum() / x_mask.sum()
            score_map["kl"] = kl_loss
            score_map["loss"] = kl_loss
            if self.training:
                kl_loss.backward()
            return score_map
        if OPTS.flowft:
            kl, mean_baseprob = self.compute_flow_KL(xz_states, yz_prob.detach(), z_mask=z_mask)
            kl_loss = kl
            score_map["wkl"] = kl / z_mask.sum() * z_mask.size(0)
            score_map["kl"] = kl_loss
            score_map["loss"] = kl_loss
            score_map["baseprob"] = mean_baseprob
            if self.training:
                kl_loss.backward()
            return score_map
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
        if OPTS.sumloss:
            denom = x.shape[0]
        else:
            denom = None
        if self._shard_size is not None and self._shard_size > 0:
            if OPTS.phase == "limity_1":
                tile = torch.arange(1, y_mask.shape[1], 2)
                # if torch.cuda.is_available():
                #     tile = tile.cuda()
                y_mask[:, tile] = 0

            loss_scores, decoder_tensors, decoder_grads = self.compute_shard_loss(
                decoder_outputs, y, y_mask, denominator=denom, ignore_first_token=False, backward=False
            )
            loss_scores["word_acc"] *= float(y_mask.shape[0]) / y_mask.sum().float()
            score_map.update(loss_scores)
        else:
            logits = self.expand(decoder_outputs)
            loss = self.compute_loss(logits, y, y_mask, denominator=denom, ignore_first_token=False)
            acc = self.compute_word_accuracy(logits, y, y_mask, ignore_first_token=False)
            score_map["loss"] = loss
            score_map["word_acc"] = acc
        score_map = self.compute_final_loss(yz_prob, xz_prob, z_mask, score_map)
        # Backward for shard loss
        if self._shard_size is not None and self._shard_size > 0 and decoder_tensors is not None:
            decoder_tensors.append(score_map["shard_loss"])
            decoder_grads.append(None)
            torch.autograd.backward(decoder_tensors, decoder_grads)
        del score_map["shard_loss"]
        return score_map

    def sample_z(self, z_prob):
        """ Return the quantized vector given probabiliy distribution over z.
        """
        if OPTS.bottleneck == "vae":
            quantized_vector = z_prob[:, :, :self.latent_size]
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
                selector = torch.arange(self.latent_size * 2)[None, :].repeat(pidx.shape[1], 1)
                if torch.cuda.is_available():
                    selector = selector.cuda()
                selector = selector + (pidx[0][:, None] * self.latent_size * 2)
                xz_prob = xz_prob[:, torch.arange(xz_prob.shape[0]), selector]
            z = self.sample_z(xz_prob)
            if OPTS.flowft:
                z = self.flow.infer(xz_states.permute(0, 2, 1), xz_prob[:, :, :self.latent_size].permute(0,2,1)*0., x_mask)
                z = z.permute(0, 2, 1)
                z = self.postz_nn(z)
            if OPTS.rupdate:
                for _ in range(1):
                    z = self.rupdate_nn.sample_deterministic(z, x_mask, xz_states, x_mask)
                    z = self.postz_nn(z)
            if OPTS.zrerank:
                n_samples = 100
                batch_z = self.bottleneck.sample_any_dist(xz_prob, samples=n_samples)
                batch_z = self.postz_nn(batch_z)
                scores = self.rerank_nn.score(batch_z, x_mask, xz_states.expand(n_samples, -1, -1), x_mask)
                best_index = scores.argmax()
                z = batch_z[best_index].unsqueeze(0)
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
        keys = list(state_dict.keys())
        for k in keys:
            if "xz_encoders.1" in k or "xz_encoders.2" in k or "xz_softmax.1" in k or "xz_softmax.2" in k:
                del state_dict[k]
        if OPTS.flowft or OPTS.rupdate or OPTS.zrerank or "_2" in OPTS.phase:
            strict = False
        else:
            strict = True
        super(LANMTModel, self).load_state_dict(state_dict, strict=strict)

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

