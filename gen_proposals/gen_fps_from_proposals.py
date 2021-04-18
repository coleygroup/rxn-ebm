import argparse
import logging 
import os
import sys
import time
import pickle
import gc
gc.enable()

from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm  
from pathlib import Path 
from scipy import sparse
from rdkit import RDLogger

from joblib import Parallel, delayed
import multiprocessing

sys.path.append('.')
from rxnebm.data import augmentors
from rxnebm.data.preprocess import smi_to_fp
from rxnebm.proposer import retrosim_model

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

def parse_args():
    parser = argparse.ArgumentParser("proc_proposals.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="proc_proposals")
    parser.add_argument("--cleaned_data_root", help="Path to cleaned_data folder (do not change)", type=str,
                        default=None) 
    parser.add_argument("--proposals_file_prefix", help='Prefix of 3 proposals CSV files', type=str, default=None)
    parser.add_argument("--rxn_smi_file_prefix", help="Prefix of the 3 pickle files containing the train/valid/test reaction SMILES strings (do not change)", type=str,
                        default='50k_clean_rxnsmi_noreagent_allmapped_canon') 
    parser.add_argument("--output_file_prefix", help="Prefix of 3 output files containing proposals in chosen representation", type=str)
    parser.add_argument("--helper_file_prefix", help="Prefix of helper files for count_mol_fps & mol_smi_to_fp", type=str)

    # parser.add_argument("--parallelize", help="Whether to parallelize over all available cores", action="store_true") # always parallelize
    parser.add_argument("--split_every", help="Split train .npz every N steps to prevent out of memory error", 
                        type=int)
    parser.add_argument("--proposer", help="Proposer ['retrosim', 'GLN']", type=str)
    parser.add_argument("--topk", help="No. of proposals per product SMILES for training data (not guaranteed)", type=int, default=50)
    parser.add_argument("--maxk", help="No. of proposals per product SMILES for evaluation data (not guaranteed)", type=int, default=200)
    
    # GLN arguments 
    parser.add_argument("--beam_size", help="Beam size", type=int, default=200)

    # Retrosim arguments (rxnebm/proposer/retrosim_model.py)
    parser.add_argument("--max_precedents", help="No. of similar products Retrosim should compare against", type=int, default=200)

    parser.add_argument("--representation", help='the representation to compute ["fingerprints", "rxn_smiles"]', default='fingerprints')
    # fingerprint arguments, only apply if --representation=='fingerprints'
    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=3)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=16384)
    parser.add_argument("--rxn_type", help='Fingerprint reaction type ["diff", "sep", "hybrid", "hybrid_all"]', type=str, default="hybrid_all")
    parser.add_argument("--fp_type", help='Fingerprint type ["count", "bit"]', type=str, default="count")

    return parser.parse_args()
 
def main(args):
    '''
    1a. Check if proposals CSV files already exist
    1b. If above files don't exist, and only for retrosim, generate proposals for all product SMILES in USPTO_50k, takes a while: about ~13 hours on an 8-core machine for retrosim
    2. Process all proposed SMILES into fingerprints/smiles, also takes some time: about 40 min for train, 5 min each for valid & test, for 4096-dim count, diff fingerprints
    '''
    # retrieve args 
    max_prec = args.max_precedents
    beam_size = args.beam_size
    topk = args.topk
    maxk = args.maxk
    # parallelize = args.parallelize
    rxn_smi_file_prefix = args.rxn_smi_file_prefix
    helper_file_prefix = args.helper_file_prefix
    if args.split_every is None:
        args.split_every = float('+inf')

    cleaned_data_root = args.cleaned_data_root 
    representation = args.representation
    radius = args.radius
    fp_size = args.fp_size
    rxn_type = args.rxn_type
    fp_type = args.fp_type
    dtype = 'int32' # do not change 
    
    if args.output_file_prefix is None:
        if args.representation == 'fingerprints':
            output_file_prefix = f'{args.proposer}_rxn_fps_{topk}topk_{maxk}maxk_{fp_size}_{rxn_type}'
        elif args.representation == 'rxn_smiles':
            output_file_prefix = f'{args.proposer}_rxn_smiles_{topk}topk_{maxk}maxk'
        else:
            raise ValueError(f'Unrecognized args.representation {args.representation}')
    else:
        output_file_prefix = args.output_file_prefix

    if args.proposer == 'retrosim':
        # proposals_file_prefix = f"retrosim_{topk}maxtest_{max_prec}maxprec" # do not change
        proposals_file_prefix = f"retrosim_{topk}topk_{maxk}maxk_noGT" # do not change
    elif args.proposer == 'GLN_retrain':
        proposals_file_prefix = f"GLN_retrain_{topk}topk_{maxk}maxk_noGT" # do not change
    elif args.proposer == 'retroxpert':
        proposals_file_prefix = f"retroxpert_{topk}topk_{maxk}maxk_noGT" # do not change
    elif args.proposer == 'neuralsym':
        proposals_file_prefix = f"neuralsym_{topk}topk_{maxk}maxk_noGT" # do not change  
    elif args.proposer == 'union':
        assert args.proposals_file_prefix is not None # user should pass in
        proposals_file_prefix = args.proposals_file_prefix
        # proposals_file_prefix = 'GLN_retrain_50topk_200maxk_retrosim_50topk_200maxk_noGT' # hardcoded for now, TODO: make into arg
    elif args.proposer == 'MT':
        raise NotImplementedError

    if args.proposer in ['GLN_retrain', 'retroxpert', 'retrosim', 'neuralsym']: #TODO: MT
        for phase in ['train', 'valid', 'test']:
            files = os.listdir(cleaned_data_root)
            if proposals_file_prefix not in files:
                topk_pass, maxk_pass = False, False
                for filename in files:
                    if args.proposer in filename and f'noGT_train.csv' in filename and not '.csv.' in filename:
                    # if args.proposer in filename and f'{beam_size}beam_train.csv' in filename and not '.csv.' in filename: 
                        for kwarg in filename.split('_'): # split into ['GLN', '200topk', '200maxk', '200beam', 'valid.csv']
                            if 'topk' in kwarg:
                                if int(kwarg.strip('topk')) >= topk: 
                                    topk_pass = True  # as long as proposal file's topk >= requested topk, we can load it
                            if 'maxk' in kwarg:
                                if int(kwarg.strip('maxk')) >= maxk:
                                    maxk_pass = True
                    if topk_pass and maxk_pass: 
                        proposals_file_prefix = filename.replace('_train.csv', '')
                        break

            logging.info(f"finding proposals CSV file at : {cleaned_data_root / f'{proposals_file_prefix}_{phase}.csv'}")
            if not (cleaned_data_root / f'{proposals_file_prefix}_{phase}.csv').exists():
                raise RuntimeError(f'Could not find proposals CSV file by {args.proposer}! Please generate the proposals CSV file, for example by running gen_gln.py')
    
    elif args.proposer == 'union':
        pass
    else:
        raise ValueError(f'Unrecognized proposer {args.proposer}')

    # check if ALL 3 files to be computed already exist in cleaned_data_root, if so, then just exit the function
    phases_to_compute = ['train', 'valid', 'test']
    for phase in ["train", "valid", "test"]: 
        if (cleaned_data_root / f"{output_file_prefix}_{phase}.npz").exists() or \
        (cleaned_data_root / f"{output_file_prefix}_{phase}.pickle").exists():
            print(f"At: {cleaned_data_root / output_file_prefix}_{phase}")
            print("The processed proposals file already exists!")
            phases_to_compute.remove(phase)        
    if len(phases_to_compute) == 0: # all pre-computed, OK exit 
        return
    elif len(phases_to_compute) < 3: # corrupted 
        raise RuntimeError('The proposal processing is most likely corrupted. Please delete existing files and restart!')
    
    logging.info(f'\n{args}\n')
    
    for phase in phases_to_compute:
        logging.info(f'Processing {phase}')
        
        proposals = pd.read_csv(cleaned_data_root / f'{proposals_file_prefix}_{phase}.csv', index_col=None, dtype='str')
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

        if representation == 'rxn_smiles':
            with open(cleaned_data_root / f'{output_file_prefix}_{phase}.pickle', 'wb') as handle:
                pickle.dump(processed_rxn_smis, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'Saved {phase} rxn_smi!')

        elif representation == 'fingerprints':
            phase_rxn_fps = [] # sparse.csr_matrix((len(proposals_file_prefix), fp_size)) # init big output sparse matrix
            if phase == 'train':
                # if args.parallelize:
                def rxn_fp_helper_train(rxn, i):
                    pos_rxn_smi = rxn[0]
                    neg_rxn_smis = rxn[1:]

                    pos_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi_dist(pos_rxn_smi, fp_type, radius, fp_size, dtype)
                    pos_rxn_fp = augmentors.make_rxn_fp(pos_rcts_fp, prod_fp, rxn_type)

                    neg_rxn_fps = []
                    for neg_rxn_smi in neg_rxn_smis:
                        try:
                            neg_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi_dist(neg_rxn_smi, fp_type, radius, fp_size, dtype)
                            neg_rxn_fp = augmentors.make_rxn_fp(neg_rcts_fp, prod_fp, rxn_type)
                            neg_rxn_fps.append(neg_rxn_fp)
                        except Exception as e:
                            logging.info(f'Error {e} at index {i}')
                            continue 

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
                # else:
                #     # unparallelized version
                #     for i, rxn in enumerate(tqdm(processed_rxn_smis, desc='Generating rxn fps...')):
                #         pos_rxn_smi = rxn[0]
                #         neg_rxn_smis = rxn[1:]

                #         pos_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(pos_rxn_smi, fp_type, mol_smi_to_fp, count_mol_fps)
                #         pos_rxn_fp = augmentors.make_rxn_fp(pos_rcts_fp, prod_fp, rxn_type)

                #         neg_rxn_fps = []
                #         for neg_rxn_smi in neg_rxn_smis:
                #             try:
                #                 neg_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(neg_rxn_smi, fp_type, mol_smi_to_fp, count_mol_fps)
                #                 neg_rxn_fp = augmentors.make_rxn_fp(neg_rcts_fp, prod_fp, rxn_type)
                #                 neg_rxn_fps.append(neg_rxn_fp)
                #             except Exception as e: 
                #                 logging.info(f'Error {e} at index {i}')
                #                 continue 

                #         if len(neg_rxn_fps) < topk:
                #             if rxn_type == 'sep' or rxn_type == 'hybrid':
                #                 dummy_fp = np.zeros((1, fp_size * 2))
                #             elif rxn_type == 'hybrid_all':
                #                 dummy_fp = np.zeros((1, fp_size * 3))
                #             else: # diff 
                #                 dummy_fp = np.zeros((1, fp_size))
                #             neg_rxn_fps.extend([dummy_fp] * (topk - len(neg_rxn_fps)))

                #         this_rxn_fps = sparse.hstack([pos_rxn_fp, *neg_rxn_fps])
                #         # phase_rxn_fps[i] = sparse.hstack([pos_rxn_fp, *neg_rxn_fps]) # significantly slower! 
                #         phase_rxn_fps.append(this_rxn_fps)
                    
                    # if (i % args.split_every == 0 and i > 0) or (i == len(processed_rxn_smis) - 1):
                    #     logging.info(f'Checkpointing {i}')
                    #     phase_rxn_fps_checkpoint = sparse.vstack(phase_rxn_fps)
                    #     sparse.save_npz(cleaned_data_root / f"{output_file_prefix}_{phase}_{i}.npz", phase_rxn_fps_checkpoint)
                    #     phase_rxn_fps = [] # reset to empty list

            else: # for valid/test, we cannot assume pos_rxn_smi is inside
                # if args.parallelize:
                def rxn_fp_helper_test(rxn, i):
                    rxn_smis = rxn[1:] 

                    rxn_fps = []
                    for rxn_smi in rxn_smis:
                        try:
                            rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi_dist(rxn_smi, fp_type, radius, fp_size, dtype)
                            rxn_fp = augmentors.make_rxn_fp(rcts_fp, prod_fp, rxn_type) # log=True (take log(x+1) of rct & prodfps)
                            rxn_fps.append(rxn_fp)
                        except Exception as e:
                            logging.info(f'Error {e} at index {i}')
                            continue 

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
                # else:
                #     # unparallelized version
                #     for i, rxn in enumerate(tqdm(processed_rxn_smis, desc='Generating rxn fps...')):
                #         rxn_smis = rxn[1:] 

                #         rxn_fps = []
                #         for rxn_smi in rxn_smis:
                #             try:
                #                 rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(rxn_smi, fp_type, mol_smi_to_fp, count_mol_fps)
                #                 rxn_fp = augmentors.make_rxn_fp(rcts_fp, prod_fp, rxn_type) # log=True (take log(x+1) of rct & prodfps)
                #                 rxn_fps.append(rxn_fp)
                #             except Exception as e:
                #                 logging.info(f'Error {e} at index {i}')
                #                 continue 

                #         if len(rxn_fps) < maxk:
                #             if rxn_type == 'sep' or rxn_type == 'hybrid':
                #                 dummy_fp = np.zeros((1, fp_size * 2))
                #             elif rxn_type == 'hybrid_all':
                #                 dummy_fp = np.zeros((1, fp_size * 3))
                #             else: # diff 
                #                 dummy_fp = np.zeros((1, fp_size))
                #             rxn_fps.extend([dummy_fp] * (maxk - len(rxn_fps)))

                #         this_rxn_fps = sparse.hstack(rxn_fps)
                #         phase_rxn_fps.append(this_rxn_fps)

            phase_rxn_fps = sparse.vstack(phase_rxn_fps)
            sparse.save_npz(cleaned_data_root / f"{output_file_prefix}_{phase}.npz", phase_rxn_fps)
            del phase_rxn_fps
        
        else:
            raise ValueError(f'{representation} not supported!')

def merge_chunks(args):
    logging.info(f'Merging chunks with idxs {args.split_idxs}') 
    
    chunk_A = sparse.load_npz(args.cleaned_data_root / f"{args.output_file_prefix}_{args.phase}_{args.split_idxs[0]}.npz")
    chunk_B = sparse.load_npz(args.cleaned_data_root / f"{args.output_file_prefix}_{args.phase}_{args.split_idxs[1]}.npz")
    combined = sparse.vstack([chunk_A, chunk_B])
    del chunk_A, chunk_B
    gc.collect() 

    for idx in tqdm(args.split_idxs[2:]):
        chunk = sparse.load_npz(args.cleaned_data_root / f"{args.output_file_prefix}_{args.phase}_{idx}.npz")
        combined = sparse.vstack([combined, chunk])
        del chunk 
        gc.collect() 

    sparse.save_npz(args.cleaned_data_root / f"{args.output_file_prefix}_{args.phase}.npz", combined)
    logging.info(f'Successfully merged chunks of {args.phase}!')
    del combined
    gc.collect() 
    return

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

    if args.cleaned_data_root is None:
        args.cleaned_data_root = Path(__file__).resolve().parents[1] / 'rxnebm/data/cleaned_data'
    else:
        args.cleaned_data_root = Path(args.cleaned_data_root)

    if args.proposer == 'union':
        pass
    elif args.proposer in ['GLN', 'MT']: # only these models have beam_size as they do beam search
        args.output_file_prefix = f"{args.proposer}_rxn_fps_{args.topk}topk_{args.maxk}maxk_{args.beam_size}beam_{args.fp_size}_{args.rxn_type}"
    else:
        args.output_file_prefix = f"{args.proposer}_rxn_fps_{args.topk}topk_{args.maxk}maxk_{args.fp_size}_{args.rxn_type}"
    logging.info(f'Processing {args.proposer} proposals into {args.fp_size}-dim {args.rxn_type} fingerprints\n')
    main(args) # should take <2 min for 40k train rxn_smi w/ 200 top-k (tested on 32 cores), <20 sec for 10k valid+train

    # args.phase = 'train'
    # args.split_idxs = [0, 10000, 20000, 30000, 39713]
    # merge_chunks(args)

    logging.info('Processing finished successfully')