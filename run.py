#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This model unifies the training of decoder, latent encoder, latent predictor
"""

from __future__ import division
from __future__ import print_function

import os, sys
import time
import importlib
import torch
from torch import optim
sys.path.append(".")

from nmtlab import MTTrainer, MTDataset
from nmtlab.utils import OPTS, Vocab
from nmtlab.schedulers import TransformerScheduler, SimpleScheduler
from nmtlab.utils import is_root_node
from nmtlab.evaluation import MosesBLEUEvaluator
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser

from lib_lanmt_model import LANMTModel
from lib_rescoring import load_rescoring_transformer
from datasets import get_dataset_paths

DATA_ROOT = "./mydata"
TRAINING_MAX_TOKENS = 60

ap = ArgumentParser()
ap.add_argument("--resume", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--all", action="store_true")
ap.add_argument("--opt_dtok", default="", type=str, help="dataset token")
ap.add_argument("--opt_seed", type=int, default=3, help="random seed")

# Commmon option for both autoregressive and non-autoregressive models
ap.add_argument("--opt_batchtokens", type=int, default=4096)
ap.add_argument("--opt_hiddensz", type=int, default=512)
ap.add_argument("--opt_embedsz", type=int, default=512)
ap.add_argument("--opt_heads", type=int, default=8)
ap.add_argument("--opt_shard", type=int, default=32)

# Options for LANMT
ap.add_argument("--opt_priorl", type=int, default=6, help="layers for each z encoder")
ap.add_argument("--opt_decoderl", type=int, default=6, help="number of decoder layers")
ap.add_argument("--opt_latentdim", default=8, type=int, help="dimension of latent variables")
ap.add_argument("--opt_distill", action="store_true", help="train with knowledge distillation")
ap.add_argument("--opt_annealbudget", action="store_true", help="switch of annealing KL budget")
ap.add_argument("--opt_finetune", action="store_true", help="finetune the model without limiting KL with a budget")

# Options only for inference
ap.add_argument("--opt_Tcheat", action="store_true")
ap.add_argument("--opt_Tgibbs", type=int, default=0)
ap.add_argument("--opt_Telbo", action="store_true", help="measure ELBO during inference")
ap.add_argument("--opt_T100", action="store_true", help="decode 100 sentences")
ap.add_argument("--opt_Tnorep", action="store_true", help="")
ap.add_argument("--opt_Tsearchz", action="store_true", help="")
ap.add_argument("--opt_Tsearchlen", action="store_true", help="")
ap.add_argument("--opt_Trescore", action="store_true", help="")
ap.add_argument("--opt_Tncand", default=50, type=int)

# Paths
ap.add_argument("--model_path",
                default="{}/lanmt.pt".format(DATA_ROOT))
ap.add_argument("--result_path",
                default="{}/lanmt.result".format(DATA_ROOT))
OPTS.parse(ap)

# Determine the number of GPUs to use
horovod_installed = importlib.util.find_spec("horovod") is not None
if torch.cuda.is_available() and horovod_installed:
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    part_index = hvd.rank()
    part_num = hvd.size()
    gpu_num = hvd.size()
else:
    part_index = 0
    part_num = 1
    gpu_num = 1
if is_root_node():
    print("Running on {} GPUs".format(gpu_num))

# Get the path variables
(
    train_src_corpus,
    train_tgt_corpus,
    distilled_tgt_corpus,
    truncate_datapoints,
    test_src_corpus,
    test_tgt_corpus,
    ref_path,
    src_vocab_path,
    tgt_vocab_path,
    n_valid_per_epoch,
    training_warmsteps,
    training_maxsteps,
    pretrained_autoregressive_path
) = get_dataset_paths(DATA_ROOT, OPTS.dtok)

# Define dataset
if OPTS.distill:
    tgt_corpus = distilled_tgt_corpus
else:
    tgt_corpus = train_tgt_corpus
n_valid_samples = 5000 if OPTS.finetune else 500
if OPTS.train:
    dataset = MTDataset(
        src_corpus=train_src_corpus, tgt_corpus=tgt_corpus,
        src_vocab=src_vocab_path, tgt_vocab=tgt_vocab_path,
        batch_size=OPTS.batchtokens * gpu_num, batch_type="token",
        truncate=truncate_datapoints, max_length=TRAINING_MAX_TOKENS,
        n_valid_samples=n_valid_samples)
else:
    dataset = None

# Create the model
basic_options = dict(
    dataset=dataset,
    src_vocab_size=Vocab(src_vocab_path).size(),
    tgt_vocab_size=Vocab(tgt_vocab_path).size(),
    hidden_size=OPTS.hiddensz, embed_size=OPTS.embedsz,
    n_att_heads=OPTS.heads, shard_size=OPTS.shard,
    seed=OPTS.seed
)

lanmt_options = basic_options.copy()
lanmt_options.update(dict(
    prior_layers=OPTS.priorl, decoder_layers=OPTS.decoderl,
    latent_dim=OPTS.latentdim,
    KL_budget=0. if OPTS.finetune else 1.,
    budget_annealing=OPTS.annealbudget,
    max_train_steps=training_maxsteps
))

nmt = LANMTModel(**lanmt_options)

# Load the autoregressive model for rescoring if neccessary
if OPTS.Trescore:
    load_rescoring_transformer(basic_options, pretrained_autoregressive_path)

# Training
if OPTS.train or OPTS.all:
    # Training code
    if OPTS.finetune:
        n_valid_per_epoch = 20
        scheduler = SimpleScheduler(max_epoch=3)
    else:
        scheduler = TransformerScheduler(warm_steps=training_warmsteps, max_steps=training_maxsteps)
    optimizer = optim.Adam(nmt.parameters(), lr=0.0001, betas=(0.9, 0.98))
    trainer = MTTrainer(
        nmt, dataset, optimizer,
        scheduler=scheduler, multigpu=gpu_num > 1,
        using_horovod=horovod_installed)
    OPTS.trainer = trainer
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria="loss",
    )
    if OPTS.finetune:
        pretrain_path = OPTS.model_path.replace("_klft", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
    if OPTS.resume:
        trainer.load()
    trainer.run()

# Translation
if OPTS.test or OPTS.all:
    import torch
    import horovod.torch as hvd
    torch.cuda.set_device(hvd.local_rank())
    torch.manual_seed(OPTS.seed)
    assert os.path.exists(OPTS.model_path)
    nmt.load(OPTS.model_path)
    if torch.cuda.is_available():
        nmt.cuda()
    nmt.train(False)
    if OPTS.Telbo:
        elbo_map = defaultdict(list)
    src_vocab = Vocab(src_vocab_path)
    tgt_vocab = Vocab(tgt_vocab_path)
    result_path = OPTS.result_path
    if not is_root_node():
        sys.exit()
    lines = open(test_src_corpus).readlines()
    tgt_lines = open(test_tgt_corpus).readlines()
    # Parallel translation
    gibbs_map = defaultdict(int)
    length_hits = []
    target_hits = []
    decode_times = []
    with open(OPTS.result_path, "w") as outf:
        for i, line in enumerate(lines):
            tokens = src_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x = torch.tensor([tokens])
            if torch.cuda.is_available():
                x = x.cuda()
            start_time = time.time()
            if OPTS.Tgradec:
                targets = nmt.translate(x)
                if OPTS.debug:
                    target = targets.cpu().numpy()[0].tolist()
                    print(" ".join(tgt_vocab.decode(target)[1:-1]), len(tgt_vocab.decode(target)[1:-1]))
            else:
                with torch.no_grad():
                    # Sample from prior
                    targets, lens, z, xz_states = nmt.translate(x)
                    target = targets.cpu().numpy()[0].tolist()
                    if OPTS.debug:
                        init_len = len(tgt_vocab.decode(target)[1:-1])
                        print(" ".join(tgt_vocab.decode(target)[1:-1]), init_len)
                    if OPTS.Telbo:
                        # Record EBLO
                        elbo = nmt.measure_ELBO(x, targets)
                        elbo_map[-1].append(elbo.cpu().numpy())
                    for i_run in range(OPTS.Tgibbs):
                        prev_target = tuple(target)
                        prev_z = z
                        z, _ = nmt.compute_Q(x, targets)
                        targets, _, _, _ = nmt.translate(x, latent=z, prior_states=xz_states)
                        target = targets[0].cpu().numpy().tolist()
                        cur_target = tuple(target)
                        if OPTS.Telbo:
                            # Record EBLO
                            elbo = nmt.measure_ELBO(x, targets)
                            elbo_map[i_run].append(elbo.cpu().numpy())
                        if cur_target == prev_target and not OPTS.Telbo:
                            gibbs_map[i_run + 1] += 1
                            break
                        elif i_run == OPTS.Tgibbs - 1:
                            gibbs_map[i_run + 2] += 1
                            break
            if targets is None:
                target = [2, 2, 2]
            elif OPTS.Trescore:
                scores = OPTS.teacher(x, targets)
                target = targets[scores.argmax()]
            else:
                target = targets.cpu().numpy()[0].tolist()

            end_time = time.time()
            decode_times.append((end_time - start_time) * 1000.)

            target = [t for t in target if t > 2]
            target_words = tgt_vocab.decode(target)
            target_sent = " ".join(target_words)
            outf.write(target_sent + "\n")
            sys.stdout.write(".")
            sys.stdout.flush()
    sys.stdout.write("\n")
    print(gibbs_map)
    print("decoding time: {:.0f} / {:.0f}".format(np.mean(decode_times), np.std(decode_times)))
    if length_hits:
        length_acc = np.array(length_hits).mean()
        print("length prediction accuracy: {}".format(length_acc))
    if target_hits:
        target_acc = np.mean(target_hits)
        print("target prediction accuracy:", target_acc)
    if OPTS.Telbo:
        for k in elbo_map:
            print("elbomap", k, len(elbo_map[k]), "std=", np.std(elbo_map[k]))
            elbo_map[k] = np.mean(elbo_map[k])
        print(elbo_map)

# Evaluation of translaton quality
if OPTS.evaluate or OPTS.all:
    # post-process
    if is_root_node():
        hyp_path = "/tmp/namt_hyp.txt"
        result_path = OPTS.result_path
        if OPTS.Tnorep and not os.path.exists(result_path):
            result_path = result_path.replace("_Tnorep", "")
        with open(hyp_path, "w") as outf:
            for line in open(result_path):
                if OPTS.Tnorep:
                    tokens = line.strip().split()
                    new_tokens = []
                    for tok in tokens:
                        if len(new_tokens) > 0 and tok != new_tokens[-1]:
                            new_tokens.append(tok)
                        elif len(new_tokens) == 0:
                            new_tokens.append(tok)
                    new_line = " ".join(new_tokens) + "\n"
                    line = new_line
                line = line.replace("@@ ", "")
                if "▁" in line:
                    line = line.strip()
                    line = "".join(line.split())
                    line = line.replace("▁", " ").strip() + "\n"
                outf.write(line)
        # Get BLEU score
        if "wmt" in OPTS.dtok:
            from nmtlab.evaluation import SacreBLEUEvaluator
            evaluator = SacreBLEUEvaluator(ref_path=ref_path, tokenizer="intl", lowercase=True)
            bleu = evaluator.evaluate(hyp_path)
        else:
            evaluator = MosesBLEUEvaluator(ref_path=ref_path)
            bleu = evaluator.evaluate(hyp_path)
        print("BLEU=", bleu)

