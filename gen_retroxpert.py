# import pickle
import csv
import sys
import logging
import argparse
import os
from datetime import datetime

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm

from rdkit import RDLogger
import rdkit.Chem as Chem

def parse_args():
    parser = argparse.ArgumentParser("gen_retroxpert.py")
    parser.add_argument('-f') # filler for COLAB
    
    parser.add_argument("--log_file", help="log_file", type=str, default="gen_retroxpert")
    parser.add_argument("--data_folder", help="data folder", type=str)
    parser.add_argument("--csv_prefix", help="file prefix of CSV file from retroxpert", type=str,
                        default="retroxpert_top50_max200_raw_")

    parser.add_argument("--phases", help="Phases to process", 
                        type=str, nargs='+', default=['train', 'valid', 'test'])

    parser.add_argument("--beam_size", help="Beam size", type=int, default=200)
    parser.add_argument("--topk", help="How many top-k proposals to put in train (not guaranteed)", type=int, default=50)
    parser.add_argument("--maxk", help="How many top-k proposals to generate and put in valid/test (not guaranteed)", type=int, default=200)
    return parser.parse_args()

# TODO: incorporate model scores
# if canonical_prediction_{j} is '', then skip that corresponding index in the scores np array (no significance)
# save as tuple
def process_csv(
    topk: int = 50,
    maxk: int = 200,
    beam_size: int = 200,
    phases: Optional[List[str]] = ['train', 'valid', 'test'],
    data_folder: Optional[os.PathLike] = Path('rxnebm/data/cleaned_data/'),
    csv_prefix: Optional[str] = 'retroxpert_top50_max200_beam200_raw'
):
    for phase in phases:
        logging.info(f'Processing {phase} of {phases}')
        phase_rows = []
        phase_ranks = []
        phase_topk = topk if phase == 'train' else maxk
        dup_count = 0
        with open(data_folder / f"{csv_prefix}_{phase}.csv", "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            if phase == 'train':
                for i, row in enumerate(tqdm(csv_reader)):
                    p_smi = row["product"]
                    r_smi_true = row["target"]
                    # if str(row["rank"]) == '9999' or r_smi_true == 'CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.N[C@@H](CC1CC1)C(=O)O': #  must skip all with rank = 9999???
                    #     continue
                    # else:
                    this_row = [f"{r_smi_true}>>{p_smi}", p_smi, r_smi_true] # orig_rxn_smi
                    r_smi_true_split = row["target"].split('.')
                    r_smi_true_split.sort()
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
                                    predictions.append(cand)
                            except:
                                continue

                    seen = [] # remove duplicate predictions
                    for pred in predictions:
                        if pred not in seen:
                            seen.append(pred)
                        else:
                            dup_count += 1
                    
                    true_rank = 9999
                    for i, pred in enumerate(seen):
                        pred_split = pred.split('.')
                        pred_split.sort()
                        if pred_split == r_smi_true_split:
                            true_rank = i # rank is 0-indexed
                            break

                    this_row.append(true_rank)
                    phase_ranks.append(true_rank)

                    if len(seen) < phase_topk:
                        seen.extend(['9999'] * (phase_topk - len(seen)))
                    else:
                        seen = seen[:phase_topk]
                    this_row.extend(seen)
                    phase_rows.append(this_row)
            else:
                for i, row in enumerate(tqdm(csv_reader)):
                    p_smi = row["product"]
                    r_smi_true = row["target"]
                    this_row = [f"{r_smi_true}>>{p_smi}", p_smi, r_smi_true] # orig_rxn_smi
                    r_smi_true_split = row["target"].split('.')
                    r_smi_true_split.sort()
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
                                    if cand_split == r_smi_true_split:
                                        if not found: # cannot set true_rank here bcos later when we remove duplicates it can change if those dups are before the correct prediction (i.e. true_rank will decrease)
                                            predictions.append('.'.join(cand_split)) # add to list of predictions
                                            found = True # to avoid searching for true pred once found
                                        else:
                                            continue
                                    else:
                                        predictions.append(cand)
                            except:
                                continue

                    seen = [] # remove duplicate predictions
                    for pred in predictions:
                        if pred not in seen:
                            seen.append(pred)
                        else:
                            dup_count += 1

                    if found:
                        for i, pred in enumerate(seen):
                            pred_split = pred.split('.') # sorted already
                            if pred_split == r_smi_true_split:
                                true_rank = i # rank is 0-indexed
                                break
                    this_row.append(true_rank)
                    phase_ranks.append(true_rank)

                    if len(seen) < phase_topk:
                        seen.extend(['9999'] * (phase_topk - len(seen)))
                    else:
                        seen = seen[:phase_topk]
                    this_row.extend(seen)
                    phase_rows.append(this_row)

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
            f'retroxpert_{topk}top_{maxk}max_{beam_size}beam_{phase}.csv', 'w'
        ) as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(base_col_names) # header

            for row in tqdm(phase_rows):
                writer.writerow(row)
        
        logging.info(f'Finished {phase}')

if __name__ == '__main__':
    args = parse_args() 

    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.data_folder is None:
        args.data_folder = Path(__file__).resolve().parents[0] / 'rxnebm/data/cleaned_data/' 
    else:
        args.data_folder = Path(args.data_folder)
    logging.info(args)

    process_csv(
        topk=args.topk, 
        maxk=args.maxk,
        beam_size=args.beam_size,
        phases=args.phases,
        data_folder=args.data_folder
    )
    logging.info('Finished compiling all CSVs')