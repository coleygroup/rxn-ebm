import argparse
import gc
import logging
import os
import pickle
import sys
import time

gc.enable()

import multiprocessing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

sys.path.append('.')
from rxnebm.data import fp_utils
from rxnebm.proposer import retrosim_model

from utils import tqdm_joblib

def parse_args():
    parser = argparse.ArgumentParser("proc_proposals.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="proc_proposals")
    parser.add_argument("--root", help="Path to cleaned_data folder (shouldn't need to change)", type=str,
                        default=None) 
    parser.add_argument("--proposals_file_prefix", help='Prefix of 3 proposals CSV files', type=str, default=None)
    parser.add_argument("--output_file_prefix", help="Prefix of 3 output files to be created through this script", type=str)

    parser.add_argument("--split_every", help="Split train .npz every N steps to prevent out of memory error", 
                        type=int)
    parser.add_argument("--proposer", help="Proposer ['retrosim', 'GLN']", type=str)
    parser.add_argument("--topk", help="No. of proposals per product SMILES for training data (not guaranteed)", type=int, default=50)
    parser.add_argument("--maxk", help="No. of proposals per product SMILES for evaluation data (not guaranteed)", type=int, default=200)
    # fingerprint arguments
    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=3)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=16384)
    parser.add_argument("--rxn_type", help='Fingerprint reaction type ["diff", "sep", "hybrid", "hybrid_all"]', type=str, default="hybrid_all")

    return parser.parse_args()
 
def main(args):
    '''
    1a. Check if proposals CSV files already exist
    1b. If above files don't exist, and only for retrosim, generate proposals for all product SMILES in USPTO_50k, takes a while: about ~13 hours on an 8-core machine for retrosim
    2. Process all proposed SMILES into fingerprints/smiles, also takes some time: about 40 min for train, 5 min each for valid & test, for 4096-dim count, diff fingerprints
    '''
    # retrieve args
    topk = args.topk
    maxk = args.maxk
    if args.split_every is None:
        args.split_every = float('+inf')

    root = args.root 
    radius = args.radius
    fp_size = args.fp_size
    rxn_type = args.rxn_type
    
    if args.output_file_prefix is None:
        output_file_prefix = f'{args.proposer}_rxn_fps_{topk}topk_{maxk}maxk_{fp_size}_{rxn_type}'
    else:
        output_file_prefix = args.output_file_prefix

    if args.proposals_file_prefix:
        proposals_file_prefix = args.proposals_file_prefix
    else:
        if args.proposer in ['GLN', 'retroxpert', 'retrosim', 'neuralsym', 'megan']:
            proposals_file_prefix = f"{args.proposer}_{topk}topk_{maxk}maxk_noGT" # do not change
        elif args.proposer == 'union':
            raise ValueError('Please specify --proposals_file_prefix') # user should pass in 
        else:
            raise ValueError

        if args.proposer in ['GLN', 'retroxpert', 'retrosim', 'neuralsym', 'megan']:
            for phase in ['train', 'valid', 'test']:
                files = os.listdir(root)
                if proposals_file_prefix not in files:
                    topk_pass, maxk_pass = False, False
                    for filename in files:
                        if args.proposer in filename and f'maxk_noGT' in filename and not '.csv.' in filename and filename.find('topk') == 1: # only 1x 'topk' in filename, to avoid union csv files
                            for kwarg in filename.split('_'): # split into ['GLN', '200topk', '200maxk', 'valid.csv']
                                if 'topk' in kwarg:
                                    if int(kwarg.strip('topk')) >= topk: 
                                        topk_pass = True  # as long as proposal file's topk >= requested topk, we can load it
                                if 'maxk' in kwarg:
                                    if int(kwarg.strip('maxk')) >= maxk: # same rule for maxk
                                        maxk_pass = True
                        if topk_pass and maxk_pass:
                            proposals_file_prefix = filename.replace('_train.csv', '')
                            break

                logging.info(f"finding proposals CSV file at : {root / f'{proposals_file_prefix}_{phase}.csv'}")
                if not (root / f'{proposals_file_prefix}_{phase}.csv').exists():
                    raise RuntimeError(f'Could not find proposals CSV file by {args.proposer}! Please generate the proposals CSV file, for example by running gen_gln.py')

    # check if ALL 3 files to be computed already exist in root, if so, then just exit the function
    phases_to_compute = ['train', 'valid', 'test']
    for phase in ["train", "valid", "test"]: 
        if (root / f"{output_file_prefix}_{phase}.npz").exists() or \
        (root / f"{output_file_prefix}_{phase}.pickle").exists():
            print(f"At: {root / output_file_prefix}_{phase}")
            print("The processed proposals file already exists!")
            phases_to_compute.remove(phase)        
    if len(phases_to_compute) == 0: # all pre-computed, OK exit
        return
    elif len(phases_to_compute) < 3: # corrupted 
        raise RuntimeError('The proposal processing is most likely corrupted. Please delete existing files and restart!')
    
    logging.info(f'\n{args}\n')
    
    for phase in phases_to_compute:
        logging.info(f'Processing {phase}')
        
        proposals = pd.read_csv(root / f'{proposals_file_prefix}_{phase}.csv', index_col=None, dtype='str')
        proposals = proposals.fillna('9999') 
        proposals_trimmed = proposals.drop(['orig_rxn_smi', 'rank_of_true_precursor'], axis=1)
        proposals_numpy = proposals_trimmed.values

        phase_topk = topk if phase == 'train' else maxk
        processed_rxn_smis = []
        for row in proposals_numpy:
            neg_rxn_smis = []
            pos_rxn_smi = row[1] + '>>' + row[0] # true_precursors>>prod_smi 
            for neg_precursor in row[2:][:phase_topk]: # limit to just phase_topk proposals
                if str(neg_precursor) == '9999': 
                    break # this neg_proposal and later ones in the same row are all N/A, therefore break the loop 
                else:
                    neg_rxn_smi = neg_precursor + '>>' + row[0] #neg_precursors>>prod_smi 
                    neg_rxn_smis.append(neg_rxn_smi)

            processed_rxn_smis.append([pos_rxn_smi, *neg_rxn_smis])
        logging.info('Processed into pos and neg rxn_smis!')

        phase_rxn_fps = [] # sparse.csr_matrix((len(proposals_file_prefix), fp_size)) # init big output sparse matrix
        if phase == 'train':
            # if args.parallelize:
            def rxn_fp_helper_train(rxn, i):
                pos_rxn_smi = rxn[0]
                neg_rxn_smis = rxn[1:]

                pos_rcts_fp, prod_fp = fp_utils.rcts_prod_fps_from_rxn_smi_dist(pos_rxn_smi, radius, fp_size)
                pos_rxn_fp = fp_utils.make_rxn_fp(pos_rcts_fp, prod_fp, rxn_type)

                neg_rxn_fps = []
                for neg_rxn_smi in neg_rxn_smis:
                    neg_rcts_fp, prod_fp = fp_utils.rcts_prod_fps_from_rxn_smi_dist(neg_rxn_smi, radius, fp_size)
                    neg_rxn_fp = fp_utils.make_rxn_fp(neg_rcts_fp, prod_fp, rxn_type)
                    neg_rxn_fps.append(neg_rxn_fp)

                if len(neg_rxn_fps) < topk:
                    if rxn_type == 'sep' or rxn_type == 'hybrid':
                        dummy_fp = sparse.csr_matrix((1, fp_size * 2))
                    elif rxn_type == 'hybrid_all':
                        dummy_fp = sparse.csr_matrix((1, fp_size * 3))
                    else: # diff 
                        dummy_fp = sparse.csr_matrix((1, fp_size))
                    neg_rxn_fps.extend([dummy_fp] * (topk - len(neg_rxn_fps)))

                this_rxn_fps = sparse.hstack([pos_rxn_fp, *neg_rxn_fps])
                return this_rxn_fps

            num_cores = len(os.sched_getaffinity(0))
            logging.info(f'Using {num_cores} cores')
            with tqdm_joblib(tqdm(desc="Generating rxn fps", total=len(processed_rxn_smis))) as progress_bar:
                phase_rxn_fps = Parallel(n_jobs=num_cores, max_nbytes=None)(
                                    delayed(rxn_fp_helper_train)(
                                        rxn, i
                                    ) for i, rxn in enumerate(processed_rxn_smis)
                                )

        else: # for valid/test, we cannot assume pos_rxn_smi is inside
            def rxn_fp_helper_test(rxn, i):
                rxn_smis = rxn[1:] 

                rxn_fps = []
                for rxn_smi in rxn_smis:
                    rcts_fp, prod_fp = fp_utils.rcts_prod_fps_from_rxn_smi_dist(rxn_smi, radius, fp_size)
                    rxn_fp = fp_utils.make_rxn_fp(rcts_fp, prod_fp, rxn_type) # log=True (take log(x+1) of rct & prodfps)
                    rxn_fps.append(rxn_fp)

                if len(rxn_fps) < maxk:
                    if rxn_type == 'sep' or rxn_type == 'hybrid':
                        dummy_fp = sparse.csr_matrix((1, fp_size * 2))
                    elif rxn_type == 'hybrid_all':
                        dummy_fp = sparse.csr_matrix((1, fp_size * 3))
                    else: # diff 
                        dummy_fp = sparse.csr_matrix((1, fp_size))
                    rxn_fps.extend([dummy_fp] * (maxk - len(rxn_fps)))

                this_rxn_fps = sparse.hstack(rxn_fps)
                return this_rxn_fps

            num_cores = len(os.sched_getaffinity(0))
            logging.info(f'Using {num_cores} cores')
            with tqdm_joblib(tqdm(desc="Generating rxn fps", total=len(processed_rxn_smis))) as progress_bar:
                phase_rxn_fps = Parallel(n_jobs=num_cores)(
                                    delayed(rxn_fp_helper_test)(
                                        rxn, i,
                                    ) for i, rxn in enumerate(processed_rxn_smis)
                                )

        phase_rxn_fps = sparse.vstack(phase_rxn_fps)
        sparse.save_npz(root / f"{output_file_prefix}_{phase}.npz", phase_rxn_fps)
        del phase_rxn_fps
    
if __name__ == '__main__':
    args = parse_args()
 
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs(Path(__file__).resolve().parents[1] / "logs/make_fp", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(Path(__file__).resolve().parents[1] / f"logs/make_fp/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.root is None:
        args.root = Path(__file__).resolve().parents[1] / 'rxnebm/data/cleaned_data'
    else:
        args.root = Path(args.root)

    if args.proposer == 'union' or args.output_file_prefix:
        pass
    else:
        args.output_file_prefix = f"{args.proposer}_rxn_fps_{args.topk}topk_{args.maxk}maxk_{args.fp_size}_{args.rxn_type}"

    logging.info(f'Processing {args.proposer} proposals into {args.fp_size}-dim {args.rxn_type} fingerprints\n')
    main(args) # should take <2 min for 40k train rxn_smi w/ 200 top-k (tested on 32 cores), <20 sec for 10k valid+train
    logging.info('Processing finished successfully')
