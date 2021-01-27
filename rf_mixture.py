import logging
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import date, datetime
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from typing import Dict, List, Optional, Union

from rdkit import RDLogger
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (precision_recall_curve, f1_score, auc,
                            log_loss, accuracy_score, confusion_matrix)

from rxnebm.experiment import expt_utils

def parse_args():
    parser = argparse.ArgumentParser("rf_mixture.py")
    parser.add_argument('-f') # filler for COLAB

    # mode & metadata
    parser.add_argument("--checkpoint_folder", help="checkpoint folder",
                        type=str, default=expt_utils.setup_paths("LOCAL"))
    parser.add_argument("--expt_name", help="experiment name", type=str, default="")
    parser.add_argument("--date_trained", help="date trained (DD_MM_YYYY)", 
                        type=str, default=date.today().strftime("%d_%m_%Y"))
    # parser.add_argument("--parallelize", help="Whether to parallelize over all available cores", action='store_true')
    # files
    # parser.add_argument("--location", help="location of script ['COLAB', 'LOCAL']", type=str, default="LOCAL")
    parser.add_argument("--log_file", help="log_file", type=str, default="rf_mixture")
    parser.add_argument("--prodfps_file_prefix",
                        help="npz file of product fingerprints",
                        type=str)
    parser.add_argument("--labels_file_prefix",
                        help="npy file of labels for mixture of experts",
                        type=str)
    parser.add_argument("--proposals_csv_file_prefix",
                        help="do not change (CSV file containing proposals from retro models)", type=str)
    # training params
    parser.add_argument("--checkpoint", help="whether to save model checkpoints", action="store_true")
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)
    # model params
    parser.add_argument("--n_estimators", help="Number of trees", type=int, default=600)
    parser.add_argument("--max_depth", help="Max depth of tree", type=int)
    parser.add_argument("--class_weight", help="Class weight ['balanced', 'balanced_subsample', None]", 
                        type=str)
    parser.add_argument("--verbose", help="Verbosity level", type=int, default=0)
    return parser.parse_args()

def main(args):
    # load train & test prodfps & labels
    # (retro model validation + retro model testing data, no early stopping/validation data)
    data_root = Path(__file__).resolve().parents[0] / "rxnebm" / "data" / "cleaned_data"

    prodfps_train = sparse.load_npz(data_root / f"{args.prodfps_file_prefix}_valid.npz")
    prodfps_train = prodfps_train.tocsr().toarray()
    prodfps_test = sparse.load_npz(data_root / f"{args.prodfps_file_prefix}_test.npz")
    prodfps_test = prodfps_test.tocsr().toarray()

    labels_train = np.load(data_root / f"{args.labels_file_prefix}_valid.npy")
    labels_test = np.load(data_root / f"{args.labels_file_prefix}_test.npy")

    # setup model: RF => MultiOutput
    rf = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                n_jobs=-1,
                random_state=args.random_seed,
                class_weight=args.class_weight,
                verbose=args.verbose
            )
    # param grid for random search / bayesian optimization
    # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20, endpoint=False)]
    # criterion = ['gini', 'entropy']
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt', 'log2', None]
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(5, 110, num = 16)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 3, 4, 5, 6, 7, 8, 10, 12]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 3, 4, 5]
    # # Method of selecting samples for training each tree
    # bootstrap = [True]
    # class_weight = ['balanced', None]

    rf_multi = MultiOutputClassifier(rf, n_jobs=-1)
    rf_multi.fit(prodfps_train, labels_train)

    # predict & evaluate
    probs_test = rf_multi.predict_proba(prodfps_test)       # List of 3 x [N, 2] np arrays
    probs_train = rf_multi.predict_proba(prodfps_train)     # List of 3 x [N, 2] np arrays
    probs_hstack_test, probs_hstack_train = [], []
    for i in range(len(probs_test)):
        # for each model, take just probs of positive class (1st col, ignore 0th col)
        probs_hstack_test.append(np.expand_dims(probs_test[i][:, 1], axis=-1))
        probs_hstack_train.append(np.expand_dims(probs_train[i][:, 1], axis=-1))
    probs_test = np.hstack(probs_hstack_test)
    probs_train = np.hstack(probs_hstack_train)

    preds_test = (probs_test > 0.5).astype(float)           # [N, 3] np array of float 0 or 1
    preds_train = (probs_train > 0.5).astype(float)         # [N, 3] np array of float 0 or 1

    models = ['GLN', 'RetroSim', 'RetroXpert']
    accs_test, accs_train = {}, {}
    precision_test, precision_train = {}, {}
    recall_test, recall_train = {}, {}
    auc_test, auc_train = {}, {}
    logloss_test, logloss_train = {}, {}
    confusion_test, confusion_train = {}, {}
    # get stats for each of the 3 models
    for i in range(labels_train.shape[-1]):
        accs_test[i] = accuracy_score(labels_test[:, i], preds_test[:, i])
        accs_train[i] = accuracy_score(labels_train[:, i], preds_train[:, i])
        precision_test[i], recall_test[i], _ = precision_recall_curve(
                                                        labels_test[:, i],
                                                        probs_test[:, i]
                                                    )
        precision_train[i], recall_train[i], _ = precision_recall_curve(
                                                        labels_train[:, i],
                                                        probs_train[:, i]
                                                    )
        auc_test[i] = auc(recall_test[i], precision_test[i])
        auc_train[i] = auc(recall_train[i], precision_train[i])
        logloss_test[i] = log_loss(labels_test[:, i], probs_test[:, i])
        logloss_train[i] = log_loss(labels_train[:, i], probs_train[:, i])
        confusion_test[i] = confusion_matrix(labels_test[:, i], preds_test[:, i])
        confusion_train[i] = confusion_matrix(labels_train[:, i], preds_train[:, i])

        logging.info(f'------------ {models[i]} ------------ \
                    \nTest acc: {100 * accs_test[i]:.3f}%, \
                    \nTest AUC-PRC: {auc_test[i]:.3f}, \
                    \nTest loss: {logloss_test[i]:.4f}, \
                    \nTest confusion: \n{confusion_test[i]} \
                    \n')
        logging.info(f'------------ {models[i]} ------------ \
                    \nTrain acc ({models[i]}): {100 * accs_train[i]:.3f}%, \
                    \nTrain AUC-PRC ({models[i]}): {auc_train[i]:.3f}, \
                    \nTrain loss ({models[i]}): {logloss_train[i]:.4f} \
                    \nTrain confusion: \n{confusion_train[i]} \
                    \n')
    
    # get avg stats across all 3 models
    mean_acc_test, mean_acc_train = 0, 0
    mean_auc_test, mean_auc_train = 0, 0
    mean_logloss_test, mean_logloss_train = 0, 0
    for i in range(labels_train.shape[-1]):
        mean_acc_test += accs_test[i] / labels_train.shape[-1]
        mean_acc_train += accs_train[i] / labels_train.shape[-1]
        mean_auc_test += auc_test[i] / labels_train.shape[-1]
        mean_auc_train += auc_test[i] / labels_train.shape[-1]
        mean_logloss_test += logloss_test[i] / labels_train.shape[-1]
        mean_logloss_train += logloss_train[i] / labels_train.shape[-1]
    
    logging.info(f'Test acc mean: {100 * mean_acc_test:.3f}%, \
                \nTest AUC-PRC mean: {mean_auc_test:.3f}, \
                \nTest loss mean: {mean_logloss_test:.4f} \
                \n \
                \nTrain acc mean: {100 * mean_acc_train:.3f}%, \
                \nTrain AUC-PRC mean: {mean_auc_train:.3f}, \
                \nTrain loss mean: {mean_logloss_train:.4f} \
                \n')
    
    # TODO: randomly sample 5-10 prod_smi from test CSV & log preds & labels
    return mean_auc_test # for Bayesian Optimization

if __name__ == '__main__':
    args = parse_args()

    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/rf_mixture/", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(f"./logs/rf_mixture/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logging.info(f'Running: {args.expt_name}')
    logging.info(args)
    mean_auc_test = main(args)
    # TODO: Bayesian Optimization
    logging.info('Done')
