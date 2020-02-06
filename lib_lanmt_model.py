#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nmtlab.models.transformer import Transformer
from nmtlab.modules.transformer_modules import TransformerEmbedding
from nmtlab.modules.transformer_modules import PositionalEmbedding
from nmtlab.modules.transformer_modules import LabelSmoothingKLDivLoss
from nmtlab.utils import OPTS
from nmtlab.utils import TensorMap
from nmtlab.utils import smoothed_bleu

from lib_lanmt_modules import TransformerEncoder
from lib_lanmt_modules import TransformerCrossEncoder
from lib_lanmt_modules import LengthConverter
from lib_vae import VAEBottleneck


class LANMTModel(Transformer):

    def __init__(self,
                 prior_layers=3, decoder_layers=3,
                 q_layers=6,
                 latent_dim=8,
                 KL_budget=1., KL_weight=1.,
                 budget_annealing=True,
                 max_train_steps=100000,
                 **kwargs):
        """Create Latent-variable non-autoregressive NMT model.

        Args:
            prior_layers - number of layers in prior p(z|x)
            decoder_layers - number of layers in decoder p(y|z)
            q_layers - number of layers in approximator q(z|x,y)
            latent_dim - dimension of latent variables
            KL_budget - budget of KL divergence
            KL_weight - weight of the KL term,
            budget_annealing - whether anneal the KL budget
            max_train_steps - max training iterations
        """
        self.prior_layers = prior_layers
        self.decoder_layers = decoder_layers
        self.q_layers = q_layers
        self.latent_dim = latent_dim
        self.KL_budget = KL_budget
        self.KL_weight = KL_weight
        self.budget_annealing = budget_annealing
        self.max_train_steps = max_train_steps
        if OPTS.finetune:
            self.training_criteria = "BLEU"
        else:
            self.training_criteria = "loss"
        super(LANMTModel, self).__init__(**kwargs)

    def prepare(self):
        """Define the modules
        """
        # Embedding layers
        self.x_embed_layer = TransformerEmbedding(self._src_vocab_size, self.embed_size)
        self.y_embed_layer = TransformerEmbedding(self._tgt_vocab_size, self.embed_size)
        self.pos_embed_layer = PositionalEmbedding(self.hidden_size)
        # Length Transformation
        self.length_converter = LengthConverter()
        self.length_embed_layer = nn.Embedding(500, self.hidden_size)
        # Prior p(z|x)
        self.prior_encoder = TransformerEncoder(self.x_embed_layer, self.hidden_size, self.prior_layers)
        self.prior_prob_estimator = nn.Linear(self.hidden_size, self.latent_dim * 2)
        # Approximator q(z|x,y)
        self.q_encoder_y = TransformerEncoder(self.y_embed_layer, self.hidden_size, self.q_layers)
        self.q_encoder_xy = TransformerCrossEncoder(None, self.hidden_size, self.q_layers)
        # Decoder p(y|x,z)
        self.decoder = TransformerCrossEncoder(None, self.hidden_size, self.decoder_layers, skip_connect=True)
        # Bottleneck
        self.bottleneck = VAEBottleneck(self.hidden_size, z_size=self.latent_dim, standard_var=OPTS.zeroprior)
        self.latent2vector_nn = nn.Linear(self.latent_dim, self.hidden_size)
        # Length prediction
        self.length_predictor = nn.Linear(self.hidden_size, 100)
        # Word probability estimator
        self.expander_nn = nn.Linear(self.hidden_size, self._tgt_vocab_size)
        self.label_smooth = LabelSmoothingKLDivLoss(0.1, self._tgt_vocab_size, 0)
        self.set_stepwise_training(False)

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # if self._fp16:
        #     self.half()

    def compute_Q(self, x, y):
        """Compute the approximated posterior q(z|x,y) and sample from it.
        """
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))
        # Compute p(z|y,x) and sample z
        q_states = self.compute_Q_states(self.x_embed_layer(x), x_mask, y, y_mask)
        sampled_latent, q_prob = self.sample_from_Q(q_states, sampling=False)
        return sampled_latent, q_prob

    def compute_Q_states(self, x_states, x_mask, y, y_mask):
        """Compute the states for estimating q(z|x,y).
        """
        y_states = self.q_encoder_y(y, y_mask)
        if y.size(0) > x_states.size(0) and x_states.size(0) == 1:
            x_states = x_states.expand(y.size(0), -1, -1)
            x_mask = x_mask.expand(y.size(0), -1)
        states = self.q_encoder_xy(x_states, x_mask, y_states, y_mask)
        return states

    def sample_from_Q(self, q_states, sampling=True):
        """Estimate q(z|x,y) and sample a latent variable from it.
        """
        if OPTS.diracq:
            sampling = False
        sampled_z, q_prob = self.bottleneck(q_states, sampling=sampling)
        full_vector = self.latent2vector_nn(sampled_z)
        return full_vector, q_prob

    def compute_length_predictor_loss(self, xz_states, z, z_mask, y_mask):
        """Get the loss for length predictor.
        """
        y_lens = y_mask.sum(1) - 1
        delta = (y_lens - z_mask.sum(1) + 50.).long().clamp(0, 99)
        mean_z = ((z + xz_states) * z_mask[:, :, None]).sum(1) / z_mask.sum(1)[:, None]
        logits = self.length_predictor(mean_z)
        length_loss = F.cross_entropy(logits, delta, reduction="mean")
        length_acc = self.to_float(logits.argmax(-1) == delta).mean()
        length_scores = {
            "len_loss": length_loss,
            "len_acc": length_acc
        }
        return length_scores

    def compute_vae_KL(self, prior_prob, q_prob):
        """Compute KL divergence given two Gaussians.
        """
        mu1 = q_prob[:, :, :self.latent_dim]
        var1 = F.softplus(q_prob[:, :, self.latent_dim:])
        mu2 = prior_prob[:, :, :self.latent_dim]
        if OPTS.sigmoidvar:
            var2 = torch.sigmoid(prior_prob[:, :, self.latent_dim:])
        else:
            var2 = F.softplus(prior_prob[:, :, self.latent_dim:])
        if OPTS.pvarbound != 0.:
            var2 = torch.clamp(var2, 0, OPTS.pvarbound)
        if OPTS.diracq:
            var2 = 0.5
            kl = math.log(var2 * math.sqrt(2 * math.pi) + 1e-8) + 0.5 * ((mu1 - mu2) ** 2 / (var2 ** 2 + 1e-8))
            # kl = math.log(math.sqrt(2 * math.pi)) + 0.5 * ((mu1 - mu2) ** 2)
        else:
            kl = torch.log(var2 / (var1 + 1e-8) + 1e-8) + (
                    (torch.pow(var1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(var2, 2))) - 0.5
        kl = kl.sum(-1)
        return kl

    def convert_length(self, z, z_mask, target_lens):
        """Adjust the number of latent variables.
        """
        rc = 1. / math.sqrt(2)
        converted_vectors, _ = self.length_converter(z, target_lens, z_mask)
        pos_embed = self.pos_embed_layer(converted_vectors)
        len_embed = self.length_embed_layer(target_lens.long())
        converted_vectors = rc * converted_vectors + 0.5 * pos_embed + 0.5 * len_embed[:, None, :]
        return converted_vectors

    def convert_length_with_delta(self, z, z_mask, delta):
        """Adjust the number of latent variables with predicted delta
        """
        z = z.clone()
        z_lens = z_mask.sum(1).long()
        y_lens = z_lens + delta
        converted_vectors = self.convert_length(z, z_mask, y_lens)
        # Create target-side mask
        arange = torch.arange(y_lens.max().long())
        if torch.cuda.is_available():
            arange = arange.cuda()
        y_mask = self.to_float(arange[None, :].repeat(z.size(0), 1) < y_lens[:, None])
        return converted_vectors, y_mask, y_lens

    def deterministic_sample_from_prob(self, z_prob):
        """ Obtain the mean vectors from multi-variate normal distributions.
        """
        mean_vector = z_prob[:, :, :self.latent_dim]
        full_vector = self.latent2vector_nn(mean_vector)
        return full_vector

    def predict_length(self, prior_states, z, z_mask):
        """Predict the target length based on latent variables and source states.
        """
        mean_z = ((z + prior_states) * z_mask[:, :, None]).sum(1) / z_mask.sum(1)[:, None]
        logits = self.length_predictor(mean_z)
        delta = logits.argmax(-1) - 50
        return delta

    def compute_final_loss(self, q_prob, prior_prob, x_mask, score_map):
        """ Compute the report the loss.
        """
        kl = self.compute_vae_KL(prior_prob, q_prob)
        # Apply budgets for KL divergence: KL = max(KL, budget)
        budget_upperbound = self.KL_budget
        if self.budget_annealing:
            step = OPTS.trainer.global_step()
            if OPTS.beginanneal < 0:
                beginstep = float(self.max_train_steps / 2)
            else:
                beginstep = float(OPTS.beginanneal)
            if step > beginstep:
                max_train_steps = min(int(self.max_train_steps/2), 50000) if OPTS.fastanneal else self.max_train_steps
                rate = (float(step) - beginstep) / (max_train_steps - beginstep)
                min_budget = 0.
                budget = min_budget + (budget_upperbound - min_budget) * (1. - rate)
            else:
                budget = budget_upperbound
        else:
            budget = self.KL_budget
        score_map["KL_budget"] = torch.tensor(budget)
        # Compute KL divergence
        max_mask = self.to_float((kl - budget) > 0.)
        kl = kl * max_mask + (1. - max_mask) * budget
        kl_loss = (kl * x_mask / x_mask.shape[0]).sum()
        if OPTS.nokl:
            kl_loss *= 0.0000001
        # Report KL divergence
        score_map["kl"] = kl_loss
        # Also report the averge KL for each token
        score_map["tok_kl"] = (kl * x_mask / x_mask.sum()).sum()
        # Report cross-entropy loss
        score_map["nll"] = score_map["loss"]
        # Cross-entropy loss is *already* backproped when computing softmaxes in shards
        # So only need to compute the remaining losses and then backprop them
        remain_loss = score_map["kl"].clone() * self.KL_weight
        if "len_loss" in score_map:
            remain_loss += score_map["len_loss"]
        # Report the combined loss
        score_map["loss"] = remain_loss + score_map["nll"]
        return score_map, remain_loss

    def forward(self, x, y, sampling=False, return_code=False):
        """Model training.
        """
        score_map = {}
        x_mask = self.to_float(torch.ne(x, 0))
        y_mask = self.to_float(torch.ne(y, 0))

        # ----------- Compute prior and approximated posterior -------------#
        # Compute p(z|x)
        prior_states = self.prior_encoder(x, x_mask)
        if OPTS.zeroprior:
            prior_prob = self.standard_gaussian_dist(x.shape[0], x.shape[1])
        else:
            prior_prob = self.prior_prob_estimator(prior_states)
        # Compute q(z|x,y) and sample z
        q_states = self.compute_Q_states(self.x_embed_layer(x), x_mask, y, y_mask)
        # Sample latent variables from q(z|x,y)
        z_mask = x_mask
        sampled_z, q_prob = self.sample_from_Q(q_states)

        # -----------------  Convert the length of latents ------------------#
        # Compute length prediction loss
        length_scores = self.compute_length_predictor_loss(prior_states, sampled_z, z_mask, y_mask)
        score_map.update(length_scores)
        # Padding z to fit target states
        z_with_y_length = self.convert_length(sampled_z, z_mask, y_mask.sum(-1))

        # --------------------------  Decoder -------------------------------#
        decoder_states = self.decoder(z_with_y_length, y_mask, prior_states, x_mask)

        # --------------------------  Compute losses ------------------------#
        decoder_outputs = TensorMap({"final_states": decoder_states})
        denom = x.shape[0]
        if self._shard_size is not None and self._shard_size > 0:
            loss_scores, decoder_tensors, decoder_grads = self.compute_shard_loss(
                decoder_outputs, y, y_mask, denominator=denom, ignore_first_token=False, backward=False
            )
            loss_scores["word_acc"] *= float(y_mask.shape[0]) / self.to_float(y_mask.sum())
            score_map.update(loss_scores)
        else:
            raise SystemError("Shard size must be setted or the memory is not enough for this model.")

        score_map, remain_loss = self.compute_final_loss(q_prob, prior_prob, z_mask, score_map)
        # Report smoothed BLEU during validation
        if not torch.is_grad_enabled() and self.training_criteria == "BLEU":
            logits = self.expander_nn(decoder_outputs["final_states"])
            predictions = logits.argmax(-1)
            score_map["BLEU"] = - self.get_BLEU(predictions, y)

        # --------------------------  Bacprop gradient --------------------#
        if self._shard_size is not None and self._shard_size > 0 and decoder_tensors is not None:
            decoder_tensors.append(remain_loss)
            decoder_grads.append(None)
            torch.autograd.backward(decoder_tensors, decoder_grads)
        # if torch.isnan(score_map["loss"]) or torch.isinf(score_map["loss"]):
        #     import pdb;pdb.set_trace()
        return score_map

    def translate(self, x, latent=None, prior_states=None, refine_step=0):
        """ Testing codes.
        """
        x_mask = self.to_float(torch.ne(x, 0))
        # Compute p(z|x)
        if prior_states is None:
            prior_states = self.prior_encoder(x, x_mask)
        # Sample latent variables from prior if it's not given
        if latent is None:
            if OPTS.zeroprior:
                prior_prob = self.standard_gaussian_dist(x.shape[0], x.shape[1])
            else:
                prior_prob = self.prior_prob_estimator(prior_states)
            if not OPTS.Tlatent_search:
                if OPTS.scorenet:
                    z = prior_prob[:, :, :self.latent_dim]
                    prior_prob = OPTS.scorenet.refine(z, self.x_embed_layer(x), x_mask)
                latent = self.deterministic_sample_from_prob(prior_prob)
            else:
                latent = self.bottleneck.sample_any_dist(prior_prob, samples=OPTS.Tcandidate_num, noise_level=0.5)
                latent = self.latent2vector_nn(latent)
        # Predict length
        length_delta = self.predict_length(prior_states, latent, x_mask)
        # Adjust the number of latent
        converted_z, y_mask, y_lens = self.convert_length_with_delta(latent, x_mask, length_delta + 1)
        if converted_z.size(1) == 0:
            return None, latent, prior_prob.argmax(-1)
        # Run decoder to predict the target words
        decoder_states = self.decoder(converted_z, y_mask, prior_states, x_mask)
        logits = self.expander_nn(decoder_states)
        # Get the target predictions
        if OPTS.Tlatent_search and not OPTS.Tteacher_rescore:
            # Latent search without teacher rescoring is dangeous
            # because the generative model framework can't effeciently and correctly score hypotheses
            if refine_step == OPTS.Trefine_steps:
                # In the finally step, pick the best hypotheses
                logprobs, preds = torch.log_softmax(logits, 2).max(2)
                logprobs = (logprobs * y_mask).sum(1)  # x batch x 1
                preds = preds * y_mask.long()
                # after deterimistic refinement
                pred = preds[logprobs.argmax()].unsqueeze(0)
            else:
                # Just return all candidates
                pred = logits.argmax(-1)
                pred = pred * y_mask.long()
        else:
            pred = logits.argmax(-1)

        return pred, latent, prior_states

    def standard_gaussian_dist(self, batch_size, seq_size):
        shape = (batch_size, seq_size, self.latent_dim)
        return torch.cat([torch.zeros(shape).cuda(), torch.ones(shape).cuda() * 0.55], 2)

    def get_BLEU(self, batch_y_hat, batch_y):
        """Get the average smoothed BLEU of the predictions."""
        hyps = batch_y_hat.tolist()
        refs = batch_y.tolist()
        bleus = []
        for hyp, ref in zip(hyps, refs):
            if 2 in hyp:
                hyp = hyp[:hyp.index(2)]
            if 2 in ref:
                ref = ref[:ref.index(2)]
            hyp = hyp[1:]
            ref = ref[1:]
            bleus.append(smoothed_bleu(hyp, ref))
        return torch.tensor(np.mean(bleus) * 100.)
