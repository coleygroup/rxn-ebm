import logging
import argparse
import os
import sys
import random
import pandas as pd
import numpy as np
from datetime import date, datetime
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from typing import Dict, List, Optional, Union
from joblib import dump, load

import optuna
from rdkit import RDLogger
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (precision_recall_curve, f1_score, auc, roc_auc_score,
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
    parser.add_argument("--train_single", help="Train RF model on a single experiment", action='store_true')
    parser.add_argument("--train_optuna", help="Train RF model with Optuna for args.trials times", action='store_true')
    parser.add_argument("--n_trials", help="No. of trials for hyperparameter optimization w/ Optuna", type=int, default=40)
    parser.add_argument("--display", help="Display a few examples of predictions & labels", action="store_true")
    # files
    parser.add_argument("--log_file", help="log_file", type=str, default="rf_mixture")
    parser.add_argument("--prodfps_file_prefix",
                        help="npz file of product fingerprints",
                        type=str)
    parser.add_argument("--labels_file_prefix",
                        help="npy file of labels for mixture of experts",
                        type=str)
    parser.add_argument("--proposals_csv_file_prefix",
                        help="do not change (CSV file containing proposals from retro models)", type=str)
    # parser.add_argument("--location", help="location of script ['COLAB', 'LOCAL']", type=str, default="LOCAL")
    # training params
    parser.add_argument("--checkpoint", help="whether to save model checkpoints", action="store_true")
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)
    # random forest model params for train_single
    parser.add_argument("--n_estimators", help="Number of trees", type=int, default=600)
    parser.add_argument("--max_depth", help="Max depth of tree", type=int)
    parser.add_argument("--class_weight", help="Class weight ['balanced', 'balanced_subsample', None]", 
                        type=str)
    parser.add_argument("--max_features", help="Max features to use ['auto', 'sqrt', 'log2', None]", 
                        type=str, default='auto')
    parser.add_argument("--min_samples_split", help="Min samples needed to split a node", type=int, default=2)
    parser.add_argument("--min_samples_leaf", help="Min samples required at each leaf node", type=int, default=1)
    parser.add_argument("--verbose", help="Verbosity level", type=int, default=0)
    return parser.parse_args()

def train_optuna(args):
    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trials)

    trial = study.best_trial
    print('Number of trials: {}'.format(len(study.trials)))
    print(f'Best trial: Trial #{trial.number}')
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    
    root = Path(__file__).resolve().parents[0]
    study.trials_dataframe().to_csv(root / f"logs/rf_mixture/{args.expt_name}_{dt}_trials.csv", 
                                sep='\t', index=False)
    # rerun best trial
    objective(None, True, args, trial.params)

def objective(trial, optuna=True, args=None, best_params=None):
    if best_params is not None:
        logging.info('Rerunning best trial to save model')
    # load train & test prodfps & labels
    # (retro model validation + retro model testing data, no early stopping/validation data)
    data_root = Path(__file__).resolve().parents[0] / "rxnebm" / "data" / "cleaned_data"

    prodfps_train = sparse.load_npz(data_root / f"{prodfps_file_prefix}_valid.npz")
    prodfps_train = prodfps_train.tocsr().toarray()
    prodfps_test = sparse.load_npz(data_root / f"{prodfps_file_prefix}_test.npz")
    prodfps_test = prodfps_test.tocsr().toarray()

    labels_train = np.load(data_root / f"{labels_file_prefix}_valid.npy")
    labels_test = np.load(data_root / f"{labels_file_prefix}_test.npy")

    if best_params is None and optuna:
        n_estimators = trial.suggest_int('n_estimators', low=100, high=2000, step=100)
        # Number of features to consider at every split
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
        # Maximum number of levels in tree
        max_depth = trial.suggest_int('max_depth', low=0, high=120, step=5)
        if max_depth == 0:
            max_depth = None
        # Minimum number of samples required to split a node
        min_samples_split = trial.suggest_int('min_samples_split', low=2, high=12)
        # Minimum number of samples required at each leaf node
        min_samples_leaf = trial.suggest_int('min_samples_leaf', low=1, high=5)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
    elif best_params is None and not optuna:
        n_estimators = args.n_estimators
        max_features = args.max_features
        max_depth = args.max_depth
        if max_depth == 0:
            max_depth = None
        min_samples_split = args.min_samples_split
        min_samples_leaf = args.min_samples_leaf
        class_weight = args.class_weight
    else:
        n_estimators = best_params['n_estimators']
        max_features = best_params['max_features']
        max_depth = best_params['max_depth']
        if max_depth == 0:
            max_depth = None
        min_samples_split = best_params['min_samples_split']
        min_samples_leaf = best_params['min_samples_leaf']
        class_weight = best_params['class_weight']

    # setup model: RF => MultiOutput
    rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=random_seed,
                class_weight=class_weight,
                verbose=verbose
            )
    rf_multi = MultiOutputClassifier(rf, n_jobs=-1)
    rf_multi.fit(prodfps_train, labels_train)

    if best_params is not None or (args is not None and args.checkpoint):
        logging.info('Dumping model')
        dump(rf_multi, args.checkpoint_folder / f'{args.expt_name}_best.joblib')

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

    preds_test = (probs_test > 0.5).astype(float)           # [N, 3] np array of float 0. or 1.
    preds_train = (probs_train > 0.5).astype(float)         # [N, 3] np array of float 0. or 1.

    if best_params is not None or (args is not None and args.checkpoint):
        logging.info('Dumping probs for test & train')
        np.save(args.checkpoint_folder / f'{args.expt_name}_best_probs_test.npy', probs_test)
        np.save(args.checkpoint_folder / f'{args.expt_name}_best_probs_train.npy', probs_train)

    models = ['GLN', 'RetroSim', 'RetroXpert']
    accs_test, accs_train = {}, {}
    precision_test, precision_train = {}, {}
    recall_test, recall_train = {}, {}
    auc_prc_test, auc_prc_train = {}, {}
    auc_roc_test, auc_roc_train = {}, {}
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
        auc_prc_test[i] = auc(recall_test[i], precision_test[i])
        auc_prc_train[i] = auc(recall_train[i], precision_train[i])
        auc_roc_test[i] = roc_auc_score(labels_test[:, i], probs_test[:, i])
        auc_roc_train[i] = roc_auc_score(labels_train[:, i], probs_train[:, i])
        logloss_test[i] = log_loss(labels_test[:, i], probs_test[:, i])
        logloss_train[i] = log_loss(labels_train[:, i], probs_train[:, i])
        confusion_test[i] = confusion_matrix(labels_test[:, i], preds_test[:, i])
        confusion_train[i] = confusion_matrix(labels_train[:, i], preds_train[:, i])

        logging.info(f'------------ {models[i]} ------------ \
                    \nTest acc: {100 * accs_test[i]:.3f}%, \
                    \nTest AUC-PRC: {auc_prc_test[i]:.3f}, \
                     \nTest AUC-ROC: {auc_roc_test[i]:.3f}, \
                    \nTest loss: {logloss_test[i]:.4f}, \
                    \nTest confusion: \n{confusion_test[i]} \
                    \n')
        logging.info(f'------------ {models[i]} ------------ \
                    \nTrain acc: {100 * accs_train[i]:.3f}%, \
                    \nTrain AUC-PRC: {auc_prc_train[i]:.3f}, \
                    \nTrain AUC-ROC: {auc_roc_train[i]:.3f}, \
                    \nTrain loss: {logloss_train[i]:.4f} \
                    \nTrain confusion: \n{confusion_train[i]} \
                    \n')
    
    # get avg stats across all 3 models
    mean_acc_test, mean_acc_train = 0, 0
    mean_auc_prc_test, mean_auc_prc_train = 0, 0
    mean_auc_roc_test, mean_auc_roc_train = 0, 0
    mean_logloss_test, mean_logloss_train = 0, 0
    for i in range(labels_train.shape[-1]):
        mean_acc_test += accs_test[i] / labels_train.shape[-1]
        mean_acc_train += accs_train[i] / labels_train.shape[-1]
        mean_auc_prc_test += auc_prc_test[i] / labels_train.shape[-1]
        mean_auc_prc_train += auc_prc_train[i] / labels_train.shape[-1]
        mean_auc_roc_test += auc_roc_test[i] / labels_train.shape[-1]
        mean_auc_roc_train += auc_roc_train[i] / labels_train.shape[-1]
        mean_logloss_test += logloss_test[i] / labels_train.shape[-1]
        mean_logloss_train += logloss_train[i] / labels_train.shape[-1]
    
    logging.info(f'Test acc mean: {100 * mean_acc_test:.3f}%, \
                \nTest AUC-PRC mean: {mean_auc_prc_test:.3f}, \
                \nTest AUC-ROC mean: {mean_auc_roc_test:.3f}, \
                \nTest loss mean: {mean_logloss_test:.4f} \
                \n \
                \nTrain acc mean: {100 * mean_acc_train:.3f}%, \
                \nTrain AUC-PRC mean: {mean_auc_prc_train:.3f}, \
                \nTrain AUC-ROC mean: {mean_auc_roc_train:.3f}, \
                \nTrain loss mean: {mean_logloss_train:.4f} \
                \n')
    
    if best_params is not None or (args is not None and args.display):
        csv_test = pd.read_csv(data_root / f"{proposals_csv_file_prefix}_test.csv", index_col=None, dtype='str').values
        prod_idxs = random.sample(list(range(labels_test.shape[0])), k=10)
        for idx in prod_idxs:
            prod_smi = csv_test[idx, 0]
            prod_probs = probs_test[idx]                    # (3,)
            prod_preds = (prod_probs > 0.5).astype(float)   # (3,)
            prod_labels = labels_test[idx]                  # (3,)
            logging.info(f'\nproduct SMILES:\t\t{prod_smi}')
            logging.info(f'GLN:\t\t\tprob = {prod_probs[0]:.4f}, pred = {prod_preds[0]:.0f}, label = {prod_labels[0]:.0f}')
            logging.info(f'Retrosim:\t\tprob = {prod_probs[1]:.4f}, pred = {prod_preds[1]:.0f}, label = {prod_labels[1]:.0f}')
            logging.info(f'RetroXpert:\t\tprob = {prod_probs[2]:.4f}, pred = {prod_preds[2]:.0f}, label = {prod_labels[2]:.0f}')

    return -mean_auc_prc_test # for Bayesian Optimization

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

    # "global" variables
    verbose = args.verbose
    prodfps_file_prefix = args.prodfps_file_prefix
    labels_file_prefix = args.labels_file_prefix
    proposals_csv_file_prefix = args.proposals_csv_file_prefix
    random_seed = args.random_seed

    if args.train_single:
        _ = objective(None, False, args, None)
    elif args.train_optuna:
        train_optuna(args)
    logging.info('Done')
