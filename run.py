#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This model unifies the training of decoder, latent encoder, latent predictor
"""

from __future__ import division
from __future__ import print_function

import os, sys
import time
from torch import optim
sys.path.append(".")

from nmtlab import MTTrainer, MTDataset
from nmtlab.utils import OPTS, Vocab
from nmtlab.schedulers import TransformerScheduler
from nmtlab.utils import is_root_node
from nmtlab.evaluation import MosesBLEUEvaluator
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser

from lib_lanmt_model import LANMTModel

DATA_ROOT = "./mydata"

ap = ArgumentParser()
ap.add_argument("--resume", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--all", action="store_true")
ap.add_argument("--opt_batchtokens", type=int, default=4096)
ap.add_argument("--opt_hiddensz", type=int, default=512)
ap.add_argument("--opt_embedsz", type=int, default=512)
ap.add_argument("--opt_heads", type=int, default=8)
ap.add_argument("--opt_shard", type=int, default=32)
ap.add_argument("--opt_clipnorm", type=float, default=0)
ap.add_argument("--opt_labelsmooth", type=float, default=0)
ap.add_argument("--opt_dtok", default="aspec_je", type=str)
ap.add_argument("--opt_weightdecay", action="store_true")
ap.add_argument("--opt_nohvd", action="store_true")
ap.add_argument("--warmsteps", type=int, default=8000)
ap.add_argument("--maxsteps", type=int, default=50000)
ap.add_argument("--opt_gpus", type=int, default=1)
ap.add_argument("--opt_seed", type=int, default=3)
ap.add_argument("--opt_Tbeam", type=int, default=3)

ap.add_argument("--opt_phase", type=str, default="")
ap.add_argument("--opt_deepprior", action="store_true")
ap.add_argument("--opt_newarch", action="store_true")
ap.add_argument("--opt_newarch2", action="store_true")
ap.add_argument("--opt_wae", action="store_true")
ap.add_argument("--opt_enslenpred", action="store_true")
ap.add_argument("--opt_skipz", action="store_true", help="introduce skipping for latents, only for newarch")
ap.add_argument("--opt_fixgaus", action="store_true", help="fix reparameterization trick")
ap.add_argument("--opt_priorft", action="store_true", help="finetune prior")
ap.add_argument("--opt_mixprior", type=int, default=0, help="mixture of gaussian piror")
ap.add_argument("--opt_longtrain", action="store_true")
ap.add_argument("--opt_encl", type=int, default=3, help="layers for each z encoder")
ap.add_argument("--opt_decl", type=int, default=3, help="number of decoder layers")
ap.add_argument("--opt_bottleneck", default="vq", help="vq,vqsoft,gumbel,semhash,dense,vae")
ap.add_argument("--opt_lenconv", action="store_true")
ap.add_argument("--opt_deeplen", action="store_true") 
ap.add_argument("--opt_withkl", action="store_true")
ap.add_argument("--opt_klweight", type=int, default=100, help="kl weight x 100")
ap.add_argument("--opt_zdim", default=0, type=int)
ap.add_argument("--opt_znum", default=1, type=int)
ap.add_argument("--opt_kd", action="store_true", help="distillation")
ap.add_argument("--opt_kd2", action="store_true", help="double distillation")
ap.add_argument("--opt_klbudget", action="store_true")
ap.add_argument("--opt_annealkl", action="store_true")
ap.add_argument("--opt_flowft", action="store_true", help="finetune a flow prior")
ap.add_argument("--opt_klft", action="store_true")
ap.add_argument("--opt_sgdft", action="store_true")
ap.add_argument("--opt_origft", action="store_true")
ap.add_argument("--opt_sumloss", action="store_true", help="Taking sum in ELBO")
ap.add_argument("--opt_rupdate", action="store_true", help="update z with r(z'|z, x)")
ap.add_argument("--opt_zrerank", action="store_true", help="reranking model")
ap.add_argument("--opt_lenweight", type=int, default=1)
ap.add_argument("--opt_budgetn", default=100, type=int)
ap.add_argument("--opt_chkavg", action="store_true")
ap.add_argument("--opt_Tema", action="store_true")
ap.add_argument("--opt_Tcheat", action="store_true")
ap.add_argument("--opt_Tmeasure_reconstruct", action="store_true")
ap.add_argument("--opt_Tgibbs", type=int, default=0)
ap.add_argument("--opt_Tinterpolate", type=int, default=0)
ap.add_argument("--opt_Telbo", action="store_true", help="measure ELBO during inference")
ap.add_argument("--opt_T100", action="store_true", help="decode 100 sentences")
ap.add_argument("--opt_Twmt17", action="store_true", help="decode wmt17 dataset")
ap.add_argument("--opt_Tnorep", action="store_true", help="")
ap.add_argument("--opt_Tgentrain", action="store_true", help="")
ap.add_argument("--opt_Tsearchz", action="store_true", help="")
ap.add_argument("--opt_Tsearchlen", action="store_true", help="")
ap.add_argument("--opt_Trescore", action="store_true", help="")
ap.add_argument("--opt_Tncand", default=50, type=int)

ap.add_argument("--model_path",
                default="{}/data/mcgen/models/namt3.pt".format(os.environ["HOME"]))
ap.add_argument("--result_path",
                default="{}/data/mcgen/results/namt3.result".format(os.environ["HOME"]))
OPTS.parse(ap)


if OPTS.dtok == "aspec_je":
    DATA_ROOT = "{}/data/aspec_enja".format(os.environ["HOME"])
    train_src_corpus = "{}/aspec.ja.bpe40k".format(DATA_ROOT)
    train_tgt_corpus = "{}/aspec.case.en.bpe40k".format(DATA_ROOT)
    if OPTS.kd and not OPTS.origft:
        train_tgt_corpus = "{}/data/mcgen/processed_data/aspec_je.distill.tgt".format(os.getenv("HOME"))
        assert os.path.exists(train_tgt_corpus)
    if OPTS.kd2:
        train_tgt_corpus = "{}/data/mcgen/processed_data/namt3_annealkl_bottleneck-vae_budgetn-50_decl-6_encl-6_gpus-8_kd_klbudget_lenconv_longtrain_newarch2_sumloss_withkl_zdim-8_Tgentrain_Tgibbs-3.result".format(os.getenv("HOME"))
        assert os.path.exists(train_tgt_corpus)
    test_corpus = "{}/aspec_test.ja.bpe40k".format(DATA_ROOT)
    test_tgt_corpus = "{}/aspec_test.case.en.bpe40k".format(DATA_ROOT)
    ref_path = "{}/aspec_test.case.en".format(DATA_ROOT)
    src_vocab_path = "{}/aspec.ja.bpe40k.vocab".format(DATA_ROOT)
    tgt_vocab_path = "{}/aspec.case.en.bpe40k.vocab".format(DATA_ROOT)
    n_valid_per_epoch = 4
elif OPTS.dtok == "wmt14_ende":
    DATA_ROOT = "{}/data/wmt14_ende".format(os.environ["HOME"])
    train_src_corpus = "{}/train.en.sp".format(DATA_ROOT)
    train_tgt_corpus = "{}/train.de.sp".format(DATA_ROOT)
    if OPTS.kd and not OPTS.origft:
        train_tgt_corpus = "{}/data/mcgen/processed_data/wmt14_ende.distill.tgt".format(os.getenv("HOME"))
        assert os.path.exists(train_tgt_corpus)
    test_corpus = "{}/wmt14_deen_test.en.sp".format(DATA_ROOT)
    test_tgt_corpus = "{}/wmt14_deen_test.de.sp".format(DATA_ROOT)
    ref_path = "{}/wmt14_deen_test.de".format(DATA_ROOT)
    if OPTS.Twmt17:
        test_corpus = "{}/newstest2017-ende-src.en".format(DATA_ROOT)
        ref_path = "{}/newstest2017-ende-ref.de".format(DATA_ROOT)
    src_vocab_path = "{}/wmt14.en.sp.vocab".format(DATA_ROOT)
    tgt_vocab_path = "{}/wmt14.de.sp.vocab".format(DATA_ROOT)
    n_valid_per_epoch = 8
    OPTS.warmsteps = 4000
    OPTS.maxsteps = 100000
elif OPTS.dtok == "wmt14_deen":
    DATA_ROOT = "{}/data/wmt14_ende".format(os.environ["HOME"])
    train_src_corpus = "{}/train.de.sp".format(DATA_ROOT)
    train_tgt_corpus = "{}/train.en.sp".format(DATA_ROOT)
    if OPTS.kd and not OPTS.origft:
        train_tgt_corpus = "{}/data/mcgen/processed_data/wmt14_deen.distill.tgt".format(os.getenv("HOME"))
        assert os.path.exists(train_tgt_corpus)
    if OPTS.origft:
        train_src_corpus = "{}/data/mcgen/processed_data/wmt14_deen.distill.mix.de".format(os.getenv("HOME"))
        train_tgt_corpus = "{}/data/mcgen/processed_data/wmt14_deen.distill.mix.en".format(os.getenv("HOME"))
    test_corpus = "{}/wmt14_deen_test.de.sp".format(DATA_ROOT)
    test_tgt_corpus = "{}/wmt14_deen_test.en.sp".format(DATA_ROOT)
    ref_path = "{}/wmt14_deen_test.en".format(DATA_ROOT)
    # test_corpus = "{}/train.de.sp.500".format(DATA_ROOT)
    # test_tgt_corpus = "{}/data/mcgen/processed_data/wmt14_deen.distill.500.tgt".format(os.getenv("HOME"))
    # ref_path = test_tgt_corpus
    src_vocab_path = "{}/wmt14.de.sp.vocab".format(DATA_ROOT)
    tgt_vocab_path = "{}/wmt14.en.sp.vocab".format(DATA_ROOT)
    n_valid_per_epoch = 8
    OPTS.warmsteps = 4000
    OPTS.maxsteps = 100000

if OPTS.longtrain:
    OPTS.maxsteps *= 1.5
    OPTS.maxsteps = int(OPTS.maxsteps)

if "aspec" in OPTS.dtok:
    truncate = 1500000
else:
    truncate = None
max_length = 60

# if OPTS.batchtokens > 4096:
#     mult = OPTS.batchtokens / 4096.
#     OPTS.warmsteps = int(OPTS.warmsteps / mult)
#     OPTS.maxsteps = int(OPTS.maxsteps / mult)

n_valid_samples = 5000 if OPTS.klft else 500
if OPTS.train:
    dataset = MTDataset(
        src_corpus=train_src_corpus, tgt_corpus=train_tgt_corpus,
        src_vocab=src_vocab_path, tgt_vocab=tgt_vocab_path,
        batch_size=OPTS.batchtokens * OPTS.gpus, batch_type="token", truncate=truncate, max_length=max_length,
        n_valid_samples=n_valid_samples)
else:
    dataset = None

kwargs = dict(enc_layers=OPTS.encl, dec_layers=OPTS.decl,
        dataset=dataset, hidden_size=OPTS.hiddensz, embed_size=OPTS.embedsz,
        label_uncertainty=OPTS.labelsmooth, n_att_heads=OPTS.heads, shard_size=OPTS.shard,
        seed=OPTS.seed)

if not OPTS.train:
    from nmtlab.utils import Vocab
    kwargs["src_vocab_size"] = Vocab(src_vocab_path).size()
    kwargs["tgt_vocab_size"] = Vocab(tgt_vocab_path).size()

if OPTS.phase.startswith("joint"):
    from mcgen.lib_namt_model_jointdeep import NAMTUnifiedModelJointDeep
    nmt = NAMTUnifiedModelJointDeep(**kwargs)
elif OPTS.deepprior:
    from mcgen.lib_namt_model_deepprior import NAMTUnifiedModel_DeepPrior
    nmt = NAMTUnifiedModel_DeepPrior(**kwargs)
elif OPTS.newarch:
    from mcgen.lib_namt_model_new import NAMTUnifiedNewModel
    kwargs["latent_num"] = OPTS.znum
    nmt = NAMTUnifiedNewModel(**kwargs)
elif OPTS.newarch2:
    from mcgen.lib_namt_model_new2 import NAMTUnifiedNewModel2
    nmt = NAMTUnifiedNewModel2(**kwargs)
else:
    nmt = LANMTModel(**kwargs)

if OPTS.Trescore:
    from mcgen.lib_rescoring import RescoringTransformer
    teacher_kwargs = kwargs.copy()
    del teacher_kwargs["enc_layers"]
    del teacher_kwargs["dec_layers"]
    teacher_kwargs["num_encoders"] = 6
    teacher_kwargs["num_decoders"] = 6
    OPTS.teacher = RescoringTransformer(**teacher_kwargs)
    OPTS.teacher.cuda()
    OPTS.teacher.train(False)
    if OPTS.dtok == "aspec_je":
        pretrain_path = "{}/data/torch_nmt/models/stnmt1_decl-6_dtok-aspec_je_encl-6_gpus-8_heads-8_seed-36.pt".format(os.getenv("HOME"))
    elif OPTS.dtok == "wmt14_ende":
        pretrain_path = "{}/data/mcgen/models/transformer1_batchtokens-8192_dtok-wmt14_ende_gpus-8.pt".format(
            os.getenv("HOME")
        )
    else:
        raise NotImplementedError
    OPTS.teacher.load(pretrain_path)

if OPTS.train or OPTS.all:
    # Training code
    max_steps = OPTS.maxsteps
    if OPTS.wae or OPTS.klft:
        n_valid_per_epoch = 20
    if OPTS.klft or OPTS.origft or OPTS.priorft or OPTS.wae or OPTS.flowft:
        from nmtlab.schedulers import SimpleScheduler
        scheduler = SimpleScheduler(max_epoch=3)
    else:
        scheduler = TransformerScheduler(warm_steps=OPTS.warmsteps, max_steps=max_steps)
    weight_decay = 1e-5 if OPTS.weightdecay else 0
    if OPTS.sgdft:
        # optimizer = optim.Adagrad(nmt.parameters(), lr=0.0001)
        optimizer = optim.Adam(nmt.parameters(), lr=0.0001, weight_decay=weight_decay, betas=(0.9, 0.98))
    else:
        optimizer = optim.Adam(nmt.parameters(), lr=0.0001, weight_decay=weight_decay, betas=(0.9, 0.98))
    trainer = MTTrainer(nmt, dataset, optimizer, scheduler=scheduler, multigpu=OPTS.gpus > 1,
                        using_horovod=not OPTS.nohvd)
    OPTS.trainer = trainer
    if hasattr(nmt, "training_criteria"):
        criteria = nmt.training_criteria
        if is_root_node():
            print("setting criteria to", criteria)
    else:
        criteria = "loss"
    if OPTS.chkavg and OPTS.klft:
        checkpoint_average = 5
    else:
        checkpoint_average = 0
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria=criteria,
        clip_norm=OPTS.clipnorm,
        checkpoint_average=checkpoint_average
    )
    if OPTS.origft:
        pretrain_path = OPTS.model_path.replace("_origft", "")
        pretrain_path = pretrain_path.replace("_lenweight-2", "")
        pretrain_path = pretrain_path.replace("_lenweight-3", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
        OPTS.budgetn = 0
    if OPTS.klft and not OPTS.sgdft and not OPTS.rupdate and not OPTS.zrerank and not OPTS.priorft:
        pretrain_path = OPTS.model_path.replace("_klft", "")
        if OPTS.chkavg:
            pretrain_path = pretrain_path.replace("_chkavg", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
        OPTS.budgetn = 0
    if OPTS.klft and OPTS.sgdft:
        pretrain_path = OPTS.model_path.replace("_sgdft", "").replace("_klft", "")
        if OPTS.chkavg:
            pretrain_path = pretrain_path.replace("_chkavg", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
        OPTS.budgetn = 0
    if OPTS.wae:
        pretrain_path = OPTS.model_path.replace("_wae", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
        OPTS.budgetn = 0
    if OPTS.priorft:
        pretrain_path = OPTS.model_path.replace("_priorft", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
        OPTS.budgetn = 0
    if OPTS.flowft:
        pretrain_path = OPTS.model_path.replace("_flowft", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
        OPTS.budgetn = 0
    if OPTS.rupdate:
        pretrain_path = OPTS.model_path.replace("_rupdate", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
    if OPTS.zrerank:
        pretrain_path = OPTS.model_path.replace("_zrerank", "")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
    if OPTS.phase == "limitx_2" or OPTS.phase == "limity_2":
        pretrain_path = OPTS.model_path.replace("_2", "_1")
        assert os.path.exists(pretrain_path)
        nmt.load(pretrain_path)
    if OPTS.resume:
        trainer.load()
    trainer.run()

if OPTS.test or OPTS.all:
    import torch
    import horovod.torch as hvd
    torch.cuda.set_device(hvd.local_rank())
    torch.manual_seed(81)
    # Translation
    if OPTS.chkavg:
        chk_count = 0
        state_dict = None
        for i in range(1000):
            path = OPTS.model_path + ".chk{}".format(i)
            if os.path.exists(path):
                chkpoint = torch.load(path)["model_state"]
                if state_dict is None:
                    state_dict = chkpoint
                else:
                    for key, val in chkpoint.items():
                        state_dict[key] += val
                chk_count += 1
                if chk_count == 3:
                    break
        for key in state_dict.keys():
            state_dict[key] /= float(chk_count)
        assert state_dict is not None
        if is_root_node():
            print("Averaged {} checkpoints".format(chk_count))
        nmt.load_state_dict(state_dict)
    else:
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
    if OPTS.Tgentrain:
        test_corpus = train_src_corpus
        result_path = "/tmp/{}.{}".format(os.path.basename(result_path), hvd.local_rank())
    elif not is_root_node():
        sys.exit()
    lines = open(test_corpus).readlines()
    tgt_lines = open(test_tgt_corpus).readlines()
    # Parallel translation
    if OPTS.Tgentrain and hvd.size() > 0:
        i = hvd.local_rank()
        n = hvd.size()
        lines = lines[int(len(lines)*i/n) : int(len(lines)*(i+1)/n)]
    gibbs_map = defaultdict(int)
    length_hits = []
    target_hits = []
    decode_times = []
    if OPTS.Tmeasure_reconstruct:
        torch.cuda.manual_seed_all(30)
        total_l12 = 0.
        total_l1 = 0.
        total_l2 = 0.
    with open(OPTS.result_path, "w") as outf:
        if OPTS.debug:
            lines = lines[:500]
        if OPTS.Telbo:
            lines = lines[:100]
        if OPTS.T100:
            lines = lines[:100]
        for i, line in enumerate(lines):
            tokens = src_vocab.encode("<s> {} </s>".format(line.strip()).split())
            x = torch.tensor([tokens])
            if OPTS.Tcheat or OPTS.Tinterpolate or OPTS.Tmeasure_reconstruct:
                tgt_tokens = tgt_vocab.encode("<s> {} </s>".format(tgt_lines[i].strip()).split())
                y = torch.tensor([tgt_tokens])
            if torch.cuda.is_available():
                x = x.cuda()
                if OPTS.Tcheat or OPTS.Tinterpolate or OPTS.Tmeasure_reconstruct:
                    y = y.cuda()
            start_time = time.time()
            if OPTS.Tgradec:
                targets = nmt.translate(x)
                if OPTS.debug:
                    target = targets.cpu().numpy()[0].tolist()
                    print(" ".join(tgt_vocab.decode(target)[1:-1]), len(tgt_vocab.decode(target)[1:-1]))
            else:
                with torch.no_grad():
                    if OPTS.Tcheat:
                        targets, lens, z, xz_states = nmt.translate(x, y)
                        length_hits.append(1 if lens[0] == y.shape[1] else 0)
                        if lens[0] == y.shape[1]:
                            target_hits.append((targets == y).float().mean().cpu().numpy())
                    elif OPTS.Tmeasure_reconstruct:
                        assert isinstance(nmt, NAMTUnifiedModelJointDeep)
                        l12, l1, l2 = nmt.measure_reconstruction(x, y)
                        total_l12 += l12
                        total_l1 += l1
                        total_l2 += l2
                        continue
                    else:
                        # Sample from prior
                        targets, lens, z, xz_states = nmt.translate(x)
                    if OPTS.Tinterpolate > 0:
                        interpolate_ratio = float(OPTS.Tinterpolate) / 100
                        _, _, oracle_z, _ = nmt.translate(x, y)
                        ratio = float(OPTS.Tinterpolate) / 100.
                        z = z * (1. - ratio) + oracle_z * ratio
                        targets, _, _, _ = nmt.translate(x, q=z, xz_states=xz_states)
                    target = targets.cpu().numpy()[0].tolist()
                    if OPTS.debug:
                        init_len = len(tgt_vocab.decode(target)[1:-1])
                        # if init_len > 12:
                        #     continue
                        print(" ".join(tgt_vocab.decode(target)[1:-1]), init_len)
                    if OPTS.Telbo:
                        # Record EBLO
                        elbo = nmt.measure_ELBO(x, targets)
                        elbo_map[-1].append(elbo.cpu().numpy())
                    for i_run in range(OPTS.Tgibbs):
                        prev_target = tuple(target)
                        prev_z = z
                        z, _ = nmt.compute_Q(x, targets)
                        if OPTS.Tema and i_run != 0:
                            z = 0.2 * prev_z + 0.8 * z
                        targets, _, _, _ = nmt.translate(x, q=z, xz_states=xz_states)
                        target = targets[0].cpu().numpy().tolist()
                        cur_target = tuple(target)
                        if OPTS.debug:
                            cur_len = len(tgt_vocab.decode(target)[1:-1])
                            star = "*" if cur_len != init_len else ""
                            print(" ".join(tgt_vocab.decode(target)[1:-1]), cur_len, star)
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
    if OPTS.Tmeasure_reconstruct:
        print(total_l12 / len(lines), total_l1 / len(lines), total_l2 / len(lines))
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

