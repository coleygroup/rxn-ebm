#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

import numpy as np

def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    scores = [] # minhtoo: save scores as np array
    logger.info('Will save scores!')

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        curr_scores, _ = translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )
        scores.append(curr_scores)

    try:
        scores = np.array(scores) # [1, rxn_smi, minibatch_size]
        scores = np.vstack(scores[0]).astype(np.float)
        scores = np.exp(scores) # reverse natural logarithm
        np.save(opt.output + '.npy', scores)
        logger.info(f'Saved scores to {opt.output + ".npy"}')
    except Exception as e:
        logger.info(e)
        with open(opt.output + '.pickle', 'w') as f:
            pickle.dump(scores, f)
        logger.info(f'Failed to convert to numpy array. Saved scores to {opt.output + ".pickle"}')

def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
