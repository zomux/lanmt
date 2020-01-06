#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nmtlab import Vocab
from argparse import ArgumentParser
import sys, os

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("path", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert os.path.exists(args.path)
    vocab = Vocab()
    limit = None if args.limit == 0 else args.limit
    vocab.build(args.path, limit=limit)
    for word in vocab.get_list():
        print(word)
