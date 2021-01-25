import csv
import pickle 
import sys
import logging 
import argparse
import os
from datetime import datetime

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse

from joblib import Parallel, delayed
import multiprocessing

from rdkit import RDLogger
import rdkit.Chem as Chem

from rxnebm.data.preprocess import smi_to_fp

import joblib
import contextlib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    # https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()  

def main(args):
    for phase in args.phases:
        logging.info(f'Processing {phase} of {args.phases}')

        proposals_phase_dict = {}
        for proposer, file_prefix in zip(
            args.proposers, args.proposed_smi_file_prefixes
            ):
            if proposer == 'MT':
                raise NotImplementedError
            
            with open(args.input_folder / f'{file_prefix}_{phase}.csv', 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                csv_length = 0
                for row in csv_reader:
                    csv_length += 1

            with open(args.input_folder / f'{file_prefix}_{phase}.csv', 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                # need to build dictionary: {prod_smi: {proposer_i: true_rank_i, proposer_j: true_rank_j, etc.}}
                for row in tqdm(csv_reader, desc=f'Reading {proposer} csv', total=csv_length):
                    rxn_prod_smi = Chem.MolToSmiles(Chem.MolFromSmiles(row['prod_smi']), True)
                    if rxn_prod_smi not in proposals_phase_dict:
                        proposals_phase_dict[rxn_prod_smi] = {proposer : int(row['rank_of_true_precursor'])}
                    else:
                        proposals_phase_dict[rxn_prod_smi][proposer] = int(row['rank_of_true_precursor'])
        
        with open(args.input_folder / f'{args.rxnsmi_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmi_phase = pickle.load(handle)

        if args.parallelize:
            #TODO: parallelizing is very slow... not sure why
            def prod_smi_helper(rxn_smi, radius, fp_size, proposers):
                prod_smi_map = rxn_smi.split('>>')[-1]
                prod_mol = Chem.MolFromSmiles(prod_smi_map)
                [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
                prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
                # Sometimes stereochem takes another canonicalization... (just in case)
                prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)

                prod_fp = smi_to_fp.mol_smi_to_count_fp(prod_smi_nomap, radius, fp_size)

                this_rxn_predicted = []
                for proposer in proposers:
                    proposer_true_rank = proposals_phase_dict[prod_smi_nomap][proposer]
                    if proposer_true_rank == 0:
                        this_rxn_predicted.append(1) # yes, was predicted as top-1
                    else:
                        this_rxn_predicted.append(0) # no, was not predicted as top-1

                return prod_fp, this_rxn_predicted, prod_smi_nomap

            num_cores = len(os.sched_getaffinity(0))
            logging.info(f'Using {num_cores} cores')
            with tqdm_joblib(tqdm(desc="Processing rxn_smi", total=len(clean_rxnsmi_phase))) as progress_bar:
                results = Parallel(n_jobs=num_cores)(
                                    delayed(prod_smi_helper)(
                                        rxn_smi, args.radius, args.fp_size, args.proposers
                                    ) for rxn_smi in clean_rxnsmi_phase
                                )
                all_rxn_prod_fps, all_rxn_predicted, all_prod_smi_nomap = map(list, zip(*results))

        else:
            all_rxn_predicted = []
            all_rxn_prod_fps = []
            all_prod_smi_nomap = []
            for rxn_smi in tqdm(clean_rxnsmi_phase, desc='Processing rxn_smi'):
                prod_smi_map = rxn_smi.split('>>')[-1]
                prod_mol = Chem.MolFromSmiles(prod_smi_map)
                [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
                prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
                # Sometimes stereochem takes another canonicalization... (just in case)
                prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)
                all_prod_smi_nomap.append(prod_smi_nomap)

                prod_fp = smi_to_fp.mol_smi_to_count_fp(prod_smi_nomap, args.radius, args.fp_size)
                all_rxn_prod_fps.append(prod_fp)

                this_rxn_predicted = []
                for proposer in args.proposers:
                    proposer_true_rank = proposals_phase_dict[prod_smi_nomap][proposer]
                    if proposer_true_rank == 0:
                        this_rxn_predicted.append(1) # yes, was predicted as top-1
                    else:
                        this_rxn_predicted.append(0) # no, was not predicted as top-1
                all_rxn_predicted.append(this_rxn_predicted)

        # these are the labels for supervision, 3 separate binary labels
        all_rxn_predicted = np.array(all_rxn_predicted)
        np.save(
            args.output_folder / f"{args.output_file_prefix}_labels_{phase}",
            all_rxn_predicted
        )

        # these are the input data into the network
        all_rxn_prod_fps = sparse.vstack(all_rxn_prod_fps)
        sparse.save_npz(
            args.output_folder / f"{args.output_file_prefix}_prod_fps_{phase}.npz",
            all_rxn_prod_fps
        )

        # make CSV for interpreting predictions during evaluation/debugging purpose
        phase_rows = []
        for i, prod_smi in enumerate(all_prod_smi_nomap):
            row = [prod_smi]
            row.extend(all_rxn_predicted[i])
            phase_rows.append(row)

        base_col_names = ['prod_smi']
        for proposer in args.proposers:
            base_col_names.append(f'{proposer}_predicted')
        with open(
            args.output_folder /
            f"{args.output_file_prefix}_csv_{phase}.csv", 'w'
        ) as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(base_col_names) # header

            for row in phase_rows:
                writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser("gen_mixture.py")
    parser.add_argument('-f') # filler for COLAB

    parser.add_argument("--log_file", help="log_file", type=str, default="gen_mixture")
    parser.add_argument("--input_folder", help="input folder", type=str)
    parser.add_argument("--proposed_smi_file_prefixes", 
                        help="List of file prefixes of proposed smiles" \
                        "in same order as --proposers", 
                        type=str, nargs='+',
                        default=["GLN_retrain_200topk_200maxk_200beam", \
                            'retrosim_200maxtest_200maxprec', \
                            "retroxpert_200topk_200maxk_200beam"])
    parser.add_argument("--rxnsmi_file_prefix", help="file prefix of atom-mapped rxn smiles", type=str,
                        default="50k_clean_rxnsmi_noreagent_allmapped")
    parser.add_argument("--output_folder", help="output folder", type=str)
    parser.add_argument("--output_file_prefix", help="file prefix of output numpy array", type=str)
    parser.add_argument("--location", help="location of script ['COLAB', 'LOCAL']", type=str, default="LOCAL")
    parser.add_argument("--parallelize", help="Whether to parallelize over all available cores", action='store_true')

    parser.add_argument("--phases", help="List of phases", nargs='+', 
                        type=str, default=['train', 'valid', 'test'])
    parser.add_argument("--proposers", 
                        help="List of proposers" \
                        "in same order as --proposed_smi_file_prefixes" \
                        "['GLN_retrain', 'retrosim', 'retroxpert']", 
                        type=str, nargs='+', default=['GLN_retrain', 'retrosim', 'retroxpert'])

    parser.add_argument("--radius", help="Radius of product fingerprint", type=int, default=3)
    parser.add_argument("--fp_size", help="Length of product fingerprint", type=int, default=16384)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() 

    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/gen_mixture/", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(f"./logs/gen_mixture/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.input_folder is None:
        args.input_folder = Path(__file__).resolve().parents[0] / 'rxnebm/data/cleaned_data/' 
    else:
        args.input_folder = Path(args.input_folder)
    if args.output_folder is None:
        if args.location == 'COLAB':
            args.output_folder = Path('/content/gdrive/MyDrive/rxn_ebm/datasets/mixture/')
            os.makedirs(args.output_folder, exist_ok=True)
        else:
            args.output_folder = args.input_folder
    else:
        args.output_folder = Path(args.output_folder)

    if args.output_file_prefix is None:
        output_file_prefix = ""
        for proposer in args.proposers:
            output_file_prefix += f'{proposer}_'
        args.output_file_prefix = output_file_prefix

    logging.info(args)
    main(args)
    logging.info('Done')