import argparse
import csv
import logging
import multiprocessing
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import rdkit.Chem as Chem
from joblib import Parallel, delayed
from rdkit import RDLogger
from tqdm import tqdm

from utils import tqdm_joblib, without_rdkit_log


def parse_args():
    parser = argparse.ArgumentParser("gen_retroxpert.py")
    parser.add_argument("--log_file", help="log_file", type=str, default="gen_retroxpert")
    parser.add_argument("--data_folder", help="data folder", type=str)
    parser.add_argument("--input_csv_prefix", help="file prefix of CSV file from retroxpert", type=str,
                        default="retroxpert_raw_200topk_200maxk_200beam")
    parser.add_argument("--output_csv_prefix", help="file prefix of CSV file from retroxpert", type=str,
                        default="retroxpert_200topk_200maxk_200beam")
    parser.add_argument("--phases", help="Phases to process", 
                        type=str, nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument("--topk", help="How many top-k proposals to put in train (not guaranteed)", type=int, default=200)
    parser.add_argument("--maxk", help="How many top-k proposals to generate and put in valid/test (not guaranteed)", type=int, default=200)
    return parser.parse_args()

def process_train_helper(row, phase_topk):
    dup_count = 0
    p_smi = Chem.MolToSmiles(Chem.MolFromSmiles(row["product"]), True)
    r_smi_true_split = row["target"].split('.')
    r_smi_true_split.sort()
    r_smi_true_cano = Chem.MolToSmiles(Chem.MolFromSmiles(row["target"]), True)
    r_smi_true_cano2 = Chem.MolToSmiles(Chem.MolFromSmiles(r_smi_true_cano), True)
    r_smi_true = row['target']
    this_row = [f"{r_smi_true_cano2}>>{p_smi}", p_smi, r_smi_true_cano2] # orig_rxn_smi
    predictions = []
    for j in range(1, phase_topk + 1): # go through columns for proposed precursors
        cand = row[f"canonical_prediction_{j}"]
        if cand == '':
            continue
        else:
            try:
                cand_ = Chem.MolFromSmiles(cand)
                if cand_ is not None and cand_.GetNumAtoms() > cand.count('.') + 1:
                    # https://github.com/rdkit/rdkit/issues/2289 --> explicit valence for N exceeds 3 etc.
                    predictions.append(Chem.MolToSmiles(cand_, True))
            except:
                continue

    seen = [] # remove duplicate predictions
    for pred in predictions:
        pred = Chem.MolToSmiles(Chem.MolFromSmiles(pred), True)
        if pred not in seen:
            seen.append(pred)
        else:
            dup_count += 1

    true_rank = 9999
    for i, pred in enumerate(seen):
        pred_split = pred.split('.')
        pred_split.sort()
        if pred_split == r_smi_true_split or pred in [r_smi_true_cano, r_smi_true_cano2, r_smi_true]:
            true_rank = i # rank is 0-indexed
            seen.pop(i) # remove ground truth reactant for train
            break
    # repeat again in case there were multiple GT predictions that differently satisfy one of the 3 conditions
    for j, pred in enumerate(seen):
        pred_split = pred.split('.')
        pred_split.sort()
        if pred_split == r_smi_true_split or pred in [r_smi_true_cano, r_smi_true_cano2, r_smi_true]:
            seen.pop(j) # remove ground truth reactant for train!!!
            break
    # keep filtering
    for j, pred in enumerate(seen):
        pred_split = pred.split('.')
        pred_split.sort()
        if pred_split == r_smi_true_split or pred in [r_smi_true_cano, r_smi_true_cano2, r_smi_true]:
            seen.pop(j) # remove ground truth reactant for train!!!
            break
    # keep filtering
    for j, pred in enumerate(seen):
        pred_split = pred.split('.')
        pred_split.sort()
        if pred_split == r_smi_true_split or pred in [r_smi_true_cano, r_smi_true_cano2, r_smi_true]:
            seen.pop(j) # remove ground truth reactant for train!!!
            break

    this_row.append(true_rank)

    if len(seen) < phase_topk:
        seen.extend(['9999'] * (phase_topk - len(seen)))
    else:
        seen = seen[:phase_topk]
    this_row.extend(seen)
    return this_row, true_rank, dup_count

def process_test_helper(row, phase_topk):
    dup_count = 0
    p_smi = Chem.MolToSmiles(Chem.MolFromSmiles(row["product"]), True)
    r_smi_true_split = row["target"].split('.')
    r_smi_true_split.sort()
    r_smi_true_cano = Chem.MolToSmiles(Chem.MolFromSmiles(row["target"]), True)
    r_smi_true_cano2 = Chem.MolToSmiles(Chem.MolFromSmiles(r_smi_true_cano), True)
    r_smi_true = row['target']
    this_row = [f"{r_smi_true_cano}>>{p_smi}", p_smi, r_smi_true_cano] # orig_rxn_smi
    true_rank, found = 9999, False
    predictions = []
    for j in range(1, phase_topk + 1): # go through columns for proposed precursors
        cand = row[f"canonical_prediction_{j}"]
        if cand == '':
            continue
        else:
            try:
                cand_ = Chem.MolFromSmiles(cand)
                if cand_ is not None and cand_.GetNumAtoms() > cand.count('.') + 1:
                    # https://github.com/rdkit/rdkit/issues/2289
                    cand_split = cand.split('.')
                    cand_split.sort()
                    if cand_split == r_smi_true_split or cand in [r_smi_true_cano, r_smi_true_cano2, r_smi_true]:
                        if not found: # cannot set true_rank here bcos later when we remove duplicates it can change if those dups are before the correct prediction (i.e. true_rank will decrease)
                            predictions.append(Chem.MolToSmiles(cand_, True))
                            found = True # to avoid searching for true pred once found
                        else:
                            continue
                    else:
                        predictions.append(Chem.MolToSmiles(cand_, True))
            except:
                continue

    seen = [] # remove duplicate predictions
    for pred in predictions:
        pred = Chem.MolToSmiles(Chem.MolFromSmiles(pred), True)
        if pred not in seen:
            seen.append(pred)
        else:
            dup_count += 1

    if found:
        for i, pred in enumerate(seen):
            pred_split = pred.split('.')
            pred_split.sort()
            if pred_split == r_smi_true_split or pred in [r_smi_true_cano, r_smi_true_cano2, r_smi_true]:
                true_rank = i # rank is 0-indexed
                break
    this_row.append(true_rank)

    if len(seen) < phase_topk:
        seen.extend(['9999'] * (phase_topk - len(seen)))
    else:
        seen = seen[:phase_topk]
    this_row.extend(seen)
    return this_row, true_rank, dup_count

def process_csv(
    topk: int = 50,
    maxk: int = 200,
    input_csv_prefix: str = 'retroxpert_raw_200topk_200maxk_200beam',
    output_csv_prefix: str = 'retroxpert_200topk_200maxk_200beam',
    phases: Optional[List[str]] = ['train', 'valid', 'test'],
    data_folder: Optional[os.PathLike] = Path('../rxnebm/data/cleaned_data/')
):
    for phase in phases:
        logging.info(f'Processing {phase} of {phases}')
        phase_rows = []
        phase_ranks = []
        phase_topk = topk if phase == 'train' else maxk
        dup_count = 0
        with open(data_folder / f"{input_csv_prefix}_{phase}.csv", "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            csv_length = 0
            for row in csv_reader:
                csv_length += 1

        with open(data_folder / f"{input_csv_prefix}_{phase}.csv", "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            if phase == 'train':
                num_cores = len(os.sched_getaffinity(0))
                logging.info(f'Parallelizing over {num_cores} cores')
                with tqdm_joblib(tqdm(desc="Processing predictions", total=csv_length)) as progress_bar, without_rdkit_log() as no_rdkit_log:
                    results = Parallel(n_jobs=num_cores)(
                                    delayed(process_train_helper)(row, phase_topk) for row in csv_reader
                                )
                    phase_rows, phase_ranks, dup_count = zip(*results)
                    phase_rows, phase_ranks, dup_count = list(phase_rows), list(phase_ranks), list(dup_count)
                    dup_count = sum(dup_count)
            else:
                num_cores = len(os.sched_getaffinity(0)) # multiprocessing.cpu_count()
                logging.info(f'Parallelizing over {num_cores} cores')
                with tqdm_joblib(tqdm(desc="Processing predictions", total=csv_length)) as progress_bar, without_rdkit_log() as no_rdkit_log:
                    results = Parallel(n_jobs=num_cores)(
                                    delayed(process_test_helper)(row, phase_topk) for row in csv_reader
                                )
                    phase_rows, phase_ranks, dup_count = zip(*results)
                    phase_rows, phase_ranks, dup_count = list(phase_rows), list(phase_ranks), list(dup_count)
                    dup_count = sum(dup_count)

        dup_count /= len(phase_rows)
        logging.info(f'Avg # dups per rxn_smi: {dup_count}')

        logging.info('\n')
        for n in [1, 3, 5, 10, 20, 50, 100, 200]:
            total = float(len(phase_ranks))
            acc = sum([r+1 <= n for r in phase_ranks]) / total
            logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
        logging.info('\n')

        if phase == 'train': # true precursor has been removed from the proposals, so whatever is left are negatives
            proposed_col_names = [f'neg_precursor_{i}' for i in range(1, phase_topk + 1)]
        else: # validation/testing, we don't assume true precursor is present & we also do not remove them if present
            proposed_col_names = [f'cand_precursor_{i}' for i in range(1, phase_topk + 1)]
            # logging.info(f'len(proposed_col_names): {len(proposed_col_names)}')
        base_col_names = ['orig_rxn_smi', 'prod_smi', 'true_precursors', 'rank_of_true_precursor']
        base_col_names.extend(proposed_col_names)
        with open(
            data_folder / 
            f'{output_csv_prefix}_{phase}.csv', 'w'
        ) as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(base_col_names) # header

            for row in tqdm(phase_rows):
                writer.writerow(row)
        
        logging.info(f'Finished saving CSV file for {phase}')

        with open(
            data_folder / 
            f'{output_csv_prefix}_{phase}.csv', 'r'
        ) as processed_csv:
            csv_reader = csv.DictReader(processed_csv)
            proposed_counter = Counter()
            total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
            key_count = 0
            col_prefix = 'neg_precursor' if phase == 'train' else 'cand_precursor'
            for row in csv_reader:
                this_rxn_precs = []
                for i in range(1, phase_topk + 1):
                    prec = row[f'{col_prefix}_{i}']
                    if str(prec) == '9999':
                        break
                    else:
                        this_rxn_precs.append(prec)
                precursors_count = len(this_rxn_precs)
                total_proposed += precursors_count
                if precursors_count > max_proposed:
                    max_proposed = precursors_count
                    prod_smi_max = row['prod_smi']
                if precursors_count < min_proposed:
                    min_proposed = precursors_count
                    prod_smi_min = row['prod_smi']
                
                proposed_counter[row['prod_smi']] = precursors_count
                key_count += 1
                
            logging.info(f'Average precursors proposed per prod_smi: {total_proposed / key_count}')
            logging.info(f'Min precursors: {min_proposed} for {prod_smi_min}')
            logging.info(f'Max precursors: {max_proposed} for {prod_smi_max})')

            logging.info(f'\nMost common 20:')
            for i in proposed_counter.most_common(20):
                logging.info(f'{i}')
            logging.info(f'\nLeast common 20:')
            for i in proposed_counter.most_common()[-20:]:
                logging.info(f'{i}')        

if __name__ == '__main__':
    args = parse_args() 

    os.makedirs(Path(__file__).resolve().parents[1] / "logs/gen_retroxpert/", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(Path(__file__).resolve().parents[1] / f"logs/gen_retroxpert/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.data_folder is None:
        args.data_folder = Path(__file__).resolve().parents[1] / 'rxnebm/data/cleaned_data/' 
    else:
        args.data_folder = Path(args.data_folder)
    logging.info(args)

    process_csv(
        topk=args.topk, 
        maxk=args.maxk,
        phases=args.phases,
        data_folder=args.data_folder,
        input_csv_prefix=args.input_csv_prefix,
        output_csv_prefix=args.output_csv_prefix,
    )
    logging.info('Finished compiling all CSVs')
