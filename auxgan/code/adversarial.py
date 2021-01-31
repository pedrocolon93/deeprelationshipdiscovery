#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
import torch
import pickle
import logging
import time
import json

import scipy.stats as spst

from collections import OrderedDict

from models import build_model
from trainer_batch import Trainer
from evaluator import Evaluator

parser = argparse.ArgumentParser(description='Adversarial post-processing')

parser.add_argument("--seen_file", type=str, default="../vectors/distrib.vectors", help="Seen vectors file")
parser.add_argument("--adjusted_file", type=str, default="../vectors/ar.vectors", help="Adjusted vectors file")
parser.add_argument("--unseen_file", type=str, default="../vectors/prefix.vectors", help="Unseen vectors file")
parser.add_argument("--out_dir", type=str, default="../results/", help="Where to store experiment logs and models")
parser.add_argument("--dataset_file", type=str, default="../vocab/simlexsimverb.words", help="File with list of words from datasets")

parser.add_argument("--seed", type=int, default=3, help="Initialization seed")
parser.add_argument("--verbose", type=str, default="debug", help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
## Embeddings
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="renorm", help="Normalize embeddings before training")
## Mapping
parser.add_argument("--noise", type=bool, default=False, help="Add gaussian noise to G layers and D input")
parser.add_argument("--gen_layers", type=int, default=2, help="Generator layers")
parser.add_argument("--gen_hid_dim", type=int, default=2048, help="Generator hidden layer dimensions")
parser.add_argument("--gen_dropout", type=float, default=0.5, help="Generator dropout")
parser.add_argument("--gen_input_dropout", type=float, default=0.2, help="Generator input dropout")
parser.add_argument("--gen_lambda", type=float, default=1, help="Generator loss feedback coefficient")
parser.add_argument("--sim_loss", type=str, default="max_margin", help="Similarity loss: mse or max_margin")
parser.add_argument("--sim_margin", type=float, default=1, help="Similarity margin (for max_margin losse)")
parser.add_argument("--sim_neg", type=int, default=25, help="Similarity negative examples (for max_margin loss)")
parser.add_argument("--sim_lambda", type=float, default=1, help="Similarity loss feedback coefficient")
## Discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.5, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
## Training adversarial
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--sim_optimizer", type=str, default="sgd,lr=0.1", help="Similarity optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
## Splicing from evaluation/simlex_evaluator.py
parser.add_argument('--in_simlex_file', type=str, default='../auxgan/evaluation/simlexorig999.txt', help='Path to SimLex-999 original dataset')
parser.add_argument('--in_simverb_file', type=str, default='../auxgan/evaluation/simverb3500.txt', help='Path to SimVerb-3500 original dataset')
parser.add_argument('--in_card_file', type=str, default='../auxgan/evaluation/card_ft_vecs.txt', help='Path to CARD660 original dataset')
parser.add_argument('--in_embedding_file', type=str, default='oovtest-0_05-sgd-lr-0_1/gold_embs.txt', help='Path to SimVerb-3500 original dataset')
# parser.add_argument('--in_embedding_file', type=str, help='Path to gold embeddings files that are generated from training')

params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert os.path.isfile(params.seen_file)
assert os.path.isfile(params.adjusted_file)
if not os.path.exists(params.out_dir):
    os.makedirs(params.out_dir)
    
def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)
    
    # dump parameters
    pickle.dump(params, open(os.path.join(params.out_dir, 'params.pkl'), 'wb'))
    
    # create logger
    #logging.basicConfig(filename=os.path.join(params.out_dir, 'train.log'), level=getattr(logging, params.verbose.upper()))
    logging.basicConfig(level=getattr(logging, params.verbose.upper()))
    logging.info('============ Initialized logger ============')
    logging.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logging.info('The experiment will be stored in %s' % params.out_dir)


initialize_exp(params)
src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
evaluator = Evaluator(trainer)

"""
Learning loop for Adversarial Training
"""
logging.info('----> ADVERSARIAL TRAINING <----\n\n')


########################################
##                                    ##
## Splice in simlex_evaluator.py code ##
## so that we can do evaluation after ##
## each epoch, for 10 epochs in total ##
##                                    ##
########################################


suffix = ""
# prefix = "hr_"
prefix = "en_"

# Parse the gold truth
def parse_gold_standard(in_dataset_file):
    gt_list = []
    data = [line.split() for line in open(in_dataset_file).read().splitlines()]

    # Scan over each item in the list and make a list of pairs vs scores
    # Skip the first (description) line
    print("Reading... " + os.path.basename(in_dataset_file))
    for item in data[1:]:
        pair = []
        prefixed_item1 = prefix + item[0]
        prefixed_item2 = prefix + item[1]
        prefixed_pair = [prefixed_item1, prefixed_item2]

        pair.append(prefixed_pair)
        pair.append(item[2])
        gt_list.append(pair)

    return gt_list


# Parse the input embedding file
def parse_embedding_file(in_embedding_file):
    we_dict = {}

    with open(in_embedding_file, "r") as in_file:
        lines = in_file.readlines()

    in_file.close()
    # traverse the lines and
    print('Loading and normalizing word embeddings... ' + os.path.basename(in_embedding_file))
    # input vectors, but skip the first two lines (only numbers)
    for i in range(0, len(lines)):
        temp_list = lines[i].split()
        # print temp_list
        if temp_list[0].endswith(suffix):
            dkey = temp_list.pop(0)
            # Error with some non-standard words (dot, comma), just skip them, not necessary
            try:
                x = np.array(temp_list, dtype='double')
                norm = np.linalg.norm(x)
            except ValueError:
                continue
            we_dict[dkey] = x / norm
        else:
            continue

    return we_dict


# Do the actual evaluation in terms of Spearman
def evaluate_we_spearman(reps, gt_list):
    ## Prepare two lists for computing Spearman

    simlex_gold_list_all_included = []
    simlex_gold_list_with_excluded = []

    simlex_we_list_all_included = []
    simlex_we_list_with_excluded = []

    # 1. Excluding pairs for which the WE model does not have an entry
    # 2. Treating such pairs as 0 (zero similarity)

    counter_excluded = 0
    for item_gt in gt_list:
        # item_gt[0] -> word pair
        # item_gt[1] -> their score

        word1 = item_gt[0][0] + suffix
        word2 = item_gt[0][1] + suffix

        if (not word1 in reps) or (not word2 in reps):
            simlex_gold_list_all_included.append(item_gt[1])
            simlex_we_list_all_included.append(0.0)
            counter_excluded += 1
        else:
            score_pair = np.inner(reps[word1], reps[word2])

            simlex_gold_list_all_included.append(item_gt[1])
            simlex_we_list_all_included.append(score_pair)

            simlex_gold_list_with_excluded.append(item_gt[1])
            simlex_we_list_with_excluded.append(score_pair)

    # Evaluating two lists; gold and the one generated by our WE induction model
    rho_all, pvalue_all = spst.spearmanr(simlex_gold_list_all_included, simlex_we_list_all_included)

    print("\nRESULTS:\n")
    print("rho (ALL): " + round(rho_all, 5).__str__())
    print("p-value (ALL): " + round(pvalue_all, 11).__str__() + "\n")

    rho_ex, pvalue_ex = spst.spearmanr(simlex_gold_list_with_excluded, simlex_we_list_with_excluded)
    print("rho (EXCLUDED): " + round(rho_ex, 5).__str__())
    print("p-value (EXCLUDED): " + round(pvalue_ex, 11).__str__() + "\n")
    print("TOTAL PAIRS EXCLUDED: " + counter_excluded.__str__())


###################
##               ##
## Training Loop ##
##               ##
###################

for n_epoch in range(params.n_epochs):
    logging.info('Starting adversarial training epoch %i...' % n_epoch)
    tic = time.time()
    n_words_proc = 0
    stats = {'DIS_COSTS': [], 'GEN_COSTS' : [], 'SIM_COSTS' : []}
    for n_iter in range(0, params.epoch_size+1, params.batch_size):
        # discriminator training
        for _ in range(params.dis_steps):
            trainer.dis_step(stats)
        # mapping training (discriminator fooling)
        n_words_proc += trainer.mapping_step(stats, params)
        # similarity training
        trainer.sim_step(stats)
        # log stats
        if n_iter % 500 == 0:
            stats_str = [('DIS_COSTS', 'Discriminator loss'),
            ('GEN_COSTS', 'Generator loss'),
            ('SIM_COSTS', 'Similarity loss'),]
            stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                         for k, v in stats_str if len(stats[k]) > 0]
            stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
            logging.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

            # reset
            tic = time.time()
            n_words_proc = 0
            for k, _ in stats_str:
                del stats[k][:]

        ########################################
        ##                                    ##
        ## Splice in simlex_evaluator.py code ##
        ## so that we can do evaluation after ##
        ## each epoch, for 10 epochs in total ##
        ##                                    ##
        ########################################

        ## Evaluate every 1e+5 iterations, instead of every 1e+6 iterations
        if n_iter != 0 and n_iter % 100000 == 0:
            ## Embeddings/Discriminator evaluation
            to_log = OrderedDict({'n_epoch': n_epoch})
            evaluator.all_eval(to_log)
            evaluator.eval_dis(to_log)
            VALIDATION_METRIC = 'mean_cosine'
            ## JSON log / save best model / end of epoch
            logging.info("__log__:%s" % json.dumps(to_log))
            trainer.save_best(to_log, VALIDATION_METRIC)
            # logging.info('End of epoch %i.\n\n' % n_epoch)
            ## Update the learning rate (stop if too small)
            trainer.update_lr(to_log, VALIDATION_METRIC)

            # if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
            #     logging.info('Learning rate < 1e-6. BREAK.')
            #     break

            ########################################
            ##                                    ##
            ## Export embeddings to a text format ##
            ##                                    ##
            ########################################

            trainer.reload_best()
            # trainer.export(params)
            trainer.heldoutall(params)  # export embeddings in current training
            trainer.mapping.train()  # re-enable gradients/dropout

            ## Load test files and gold embeddings
            in_simlex_file = params.in_simlex_file
            in_simverb_file = params.in_simverb_file
            in_card_file = params.in_card_file
            in_embedding_file = params.in_embedding_file

            start = time.time()

            ## Parse the input word-embedding file
            reps = parse_embedding_file(in_embedding_file)

            ## Parse target files
            gt_simlex_list = parse_gold_standard(in_simlex_file)
            gt_simverb_list = parse_gold_standard(in_simverb_file)
            gt_card_list = parse_gold_standard(in_card_file)

            ## Evaluate
            print('Begin evaluation...')
            evaluate_we_spearman(reps, gt_simlex_list)
            evaluate_we_spearman(reps, gt_simverb_list)
            evaluate_we_spearman(reps, gt_card_list)
            print('End of evaluation')

            # ## Parse and evaluate simlexorig999
            # gt_simlex_list = parse_gold_standard(in_simlex_file)
            # evaluate_we_spearman(reps, gt_simlex_list)
            #
            # ## Parse and evaluate simverb3500
            # gt_simverb_list = parse_gold_standard(in_simverb_file)
            # evaluate_we_spearman(reps, gt_simverb_list)
            #
            # ## Parse and evaluate card_ft_vecs
            # gt_card_list = parse_gold_standard(in_card_file)
            # evaluate_we_spearman(reps, gt_card_list)

            end = time.time()
            print(f'\nEvaluation time: {round((end - start), 3).__str__()}s')

    logging.info('End of epoch %i.\n\n' % n_epoch)


# ########################################
# ##                                    ##
# ## Export embeddings to a text format ##
# ##                                    ##
# ########################################
#
# trainer.reload_best()
# # trainer.export(params)
# trainer.heldoutall(params)  # export embeddings in current training
# trainer.mapping.train()  # re-enable gradients/dropout


# ########################################
# ##                                    ##
# ## Splice in simlex_evaluator.py code ##
# ## so that we can do evaluation after ##
# ## each epoch, for 10 epochs in total ##
# ##                                    ##
# ########################################
#
# # from auxgan.evaluation.simlex_evaluator import parse_gold_standard, parse_embedding_file, evaluate_we_spearman
# # from ..evaluation.simlex_evaluator import parse_gold_standard, parse_embedding_file, evaluate_we_spearman
# #
# # import sys, os, codecs, operator, time
# # # reload(sys)
# # # sys.setdefaultencoding('utf8')
# # import numpy as np
# # import scipy.stats as spst
#
# # reps = []
# suffix = ""
# # prefix = "hr_"
# prefix = "en_"
#
#
# # Parse the gold truth
# def parse_gold_standard(in_dataset_file):
#     gt_list = []
#     data = [line.split() for line in open(in_dataset_file).read().splitlines()]
#
#     # Scan over each item in the list and make a list of pairs vs scores
#     # Skip the first (description) line
#     print("Reading... " + os.path.basename(in_dataset_file))
#     for item in data[1:]:
#         pair = []
#         prefixed_item1 = prefix + item[0]
#         prefixed_item2 = prefix + item[1]
#         prefixed_pair = [prefixed_item1, prefixed_item2]
#
#         pair.append(prefixed_pair)
#         pair.append(item[2])
#         gt_list.append(pair)
#
#     return gt_list
#
#
# # Parse the input embedding file
# def parse_embedding_file(in_embedding_file):
#     we_dict = {}
#
#     with open(in_embedding_file, "r") as in_file:
#         lines = in_file.readlines()
#
#     in_file.close()
#     # traverse the lines and
#     print('Loading and normalizing word embeddings... ' + os.path.basename(in_embedding_file))
#     # input vectors, but skip the first two lines (only numbers)
#     for i in range(0, len(lines)):
#         temp_list = lines[i].split()
#         # print temp_list
#         if temp_list[0].endswith(suffix):
#             dkey = temp_list.pop(0)
#             # Error with some non-standard words (dot, comma), just skip them, not necessary
#             try:
#                 x = np.array(temp_list, dtype='double')
#                 norm = np.linalg.norm(x)
#             except ValueError:
#                 continue
#             we_dict[dkey] = x / norm
#         else:
#             continue
#
#     return we_dict
#
#
# # Do the actual evaluation in terms of Spearman
# def evaluate_we_spearman(reps, gt_list):
#     ## Prepare two lists for computing Spearman
#
#     simlex_gold_list_all_included = []
#     simlex_gold_list_with_excluded = []
#
#     simlex_we_list_all_included = []
#     simlex_we_list_with_excluded = []
#
#     # 1. Excluding pairs for which the WE model does not have an entry
#     # 2. Treating such pairs as 0 (zero similarity)
#
#     counter_excluded = 0
#     for item_gt in gt_list:
#         # item_gt[0] -> word pair
#         # item_gt[1] -> their score
#
#         word1 = item_gt[0][0] + suffix
#         word2 = item_gt[0][1] + suffix
#
#         if (not word1 in reps) or (not word2 in reps):
#             simlex_gold_list_all_included.append(item_gt[1])
#             simlex_we_list_all_included.append(0.0)
#             counter_excluded += 1
#         else:
#             score_pair = np.inner(reps[word1], reps[word2])
#
#             simlex_gold_list_all_included.append(item_gt[1])
#             simlex_we_list_all_included.append(score_pair)
#
#             simlex_gold_list_with_excluded.append(item_gt[1])
#             simlex_we_list_with_excluded.append(score_pair)
#
#     # Evaluating two lists; gold and the one generated by our WE induction model
#     rho_all, pvalue_all = spst.spearmanr(simlex_gold_list_all_included, simlex_we_list_all_included)
#
#     print("\nRESULTS:\n")
#     print("rho (ALL): " + round(rho_all, 5).__str__())
#     print("p-value (ALL): " + round(pvalue_all, 11).__str__() + "\n")
#
#     rho_ex, pvalue_ex = spst.spearmanr(simlex_gold_list_with_excluded, simlex_we_list_with_excluded)
#     print("rho (EXCLUDED): " + round(rho_ex, 5).__str__())
#     print("p-value (EXCLUDED): " + round(pvalue_ex, 11).__str__() + "\n")
#     print("TOTAL PAIRS EXCLUDED: " + counter_excluded.__str__())


# in_simlex_file = params.in_simlex_file
# in_simverb_file = params.in_simverb_file
# in_embedding_file = params.in_embedding_file
#
# start = time.time()
#
# ## Parse the input gold-standard files
# gt_simlex_list = parse_gold_standard(in_simlex_file)
# gt_simverb_list = parse_gold_standard(in_simverb_file)
#
# ## Parse the input word-embedding file
# reps = parse_embedding_file(in_embedding_file)
#
# ## Do evaluations on learned representations
# evaluate_we_spearman(reps, gt_simlex_list)
# evaluate_we_spearman(reps, gt_simverb_list)
#
# end = time.time()
#
# print(f'\nEvaluation time: {round((end - start), 3).__str__()}s')
