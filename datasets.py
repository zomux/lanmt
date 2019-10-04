#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_dataset_paths(data_root, dataset_tok):
    if dataset_tok == "aspec_jaen":
        train_src_corpus = "{}/aspec.ja.bpe40k".format(data_root)
        train_tgt_corpus = "{}/aspec.case.en.bpe40k".format(data_root)
        distilled_tgt_corpus = "{}/aspec_je.distill.tgt".format(data_root)
        truncate_datapoints = 1500000

        test_src_corpus = "{}/aspec_test.ja.bpe40k".format(data_root)
        test_tgt_corpus = "{}/aspec_test.case.en.bpe40k".format(data_root)
        ref_path = "{}/aspec_test.case.en".format(data_root)

        src_vocab_path = "{}/aspec.ja.bpe40k.vocab".format(data_root)
        tgt_vocab_path = "{}/aspec.case.en.bpe40k.vocab".format(data_root)

        n_valid_per_epoch = 4
        training_warmsteps = 8000
        training_maxsteps = 50000
        pretrained_autoregressive_path = "{}/aspec_jaen_teacher.pt".format(data_root)
    if dataset_tok == "wmt14_ende":
        train_src_corpus = "{}/wmt14_ende_train.en.sp".format(data_root)
        train_tgt_corpus = "{}/wmt14_ende_train.de.sp".format(data_root)
        distilled_tgt_corpus = "{}/wmt14_ende.distill.tgt".format(data_root)
        truncate_datapoints = None

        test_src_corpus = "{}/wmt14_ende_test.en.sp".format(data_root)
        test_tgt_corpus = "{}/wmt14_ende_test.de.sp".format(data_root)
        ref_path = "{}/wmt14_ende_test.de".format(data_root)

        src_vocab_path = "{}/wmt14.en.sp.vocab".format(data_root)
        tgt_vocab_path = "{}/wmt14.de.sp.vocab".format(data_root)

        n_valid_per_epoch = 8
        training_warmsteps = 4000
        training_maxsteps = 100000
        pretrained_autoregressive_path = "{}/wmt14_ende_teacher.pt".format(data_root)

    return (
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
    )
