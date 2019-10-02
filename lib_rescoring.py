#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from nmtlab.models import Transformer
from nmtlab.utils import MapDict, OPTS


class RescoringTransformer(Transformer):

    def forward(self, src_seq, tgt_seq):
        with torch.no_grad():
            src_mask = torch.ne(src_seq, 0).float()
            tgt_mask = torch.ne(tgt_seq, 0).float()
            encoder_outputs = MapDict(self.encode(src_seq, src_mask))
            context, states = self.pre_decode(encoder_outputs, tgt_seq, src_mask=src_mask, tgt_mask=tgt_mask)
            decoder_outputs = self.decode(context, states)
            logits = self.expand(decoder_outputs)
            logp = torch.log_softmax(logits, 2)
            flat_logp = logp.view(-1, self._tgt_vocab_size)
            flat_tgt = tgt_seq[:, 1:].flatten()
            logp = - torch.nn.functional.nll_loss(flat_logp, flat_tgt, reduction="none")
            logp = logp.view(tgt_seq.shape[0], tgt_seq.shape[1]-1)
            scores = (logp * tgt_mask[:, 1:]).sum(1)
        return scores.cpu().numpy()


def load_rescoring_transformer(model_options, pretrained_path):
    teacher_kwargs = model_options.copy()
    teacher_kwargs["num_encoders"] = 6
    teacher_kwargs["num_decoders"] = 6
    # Register to OPTS so the model can be used anywhere
    OPTS.teacher = RescoringTransformer(**teacher_kwargs)
    OPTS.teacher.cuda()
    OPTS.teacher.train(False)
    assert os.path.exists(pretrained_path)
    OPTS.teacher.load(pretrained_path)
    return OPTS.teacher
