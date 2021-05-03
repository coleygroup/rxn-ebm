# import csv
import sys
import logging
# import time
import argparse
import os
import pickle
# import numpy as np
# import random
# import multiprocessing
import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter
# from functools import partial
from tqdm import tqdm
from rdkit import Chem, RDLogger

from utils import without_rdkit_log

# note: we do not specify beam size for megan as it might introduce confusion. For GLN and RetroXpert, the inference was done with beam_size = 200 across train, valid and test
# thus, we specified beam size for the proposal CSVs from those models
# however, for MEGAN, inference on train was not done with beam_size = 200, but rather 50, as it takes too long (>50 hours) to propose on 40k reactions with beam_size = 200, 
# and we only need top-50 proposals for training. for test and valid phases, beam_size = 200 was still used.

DATA_FOLDER = Path(__file__).resolve().parents[1] / 'rxnebm' / 'data' / 'cleaned_data'

def compile_into_csv(args):
    # almost boilerplate code, lines until dup_count /= len(clean_rxnsmi_phase) have been adapted to MEGAN
    for phase in args.phases:
        # load prediction files
        # read text pred files from megan, takes a while since the files are pretty big (600M for test/valid, 1.1G for train)
        # so this is going to need some RAM
        # alternative is to read line by line, but in the interest of time, let's make that a TODO
        with open(DATA_FOLDER / f"megan_raw_{args.topk}topk_{args.maxk}maxk_{args.seed}_{phase}.txt", 'r') as f:
            raw_preds = f.readlines()
        
        # load original mapped_rxn_smi, note that the mapped product/rcts in preds are NOT the original, as we scrambled the mapping
        # according to true RDKit canonical order. we will ignore the new mapping & remove it on-the-fly from proposals
        # megan generates atom-mapped proposals
        with open(DATA_FOLDER / f'{args.rxn_smi_prefix}_{phase}.pickle', 'rb') as f:
            clean_rxnsmi_phase = pickle.load(f)

        proposals_phase = {} # helper for analyse_proposed() 
        proposed_precs_phase, prod_smiles_phase, rcts_smiles_phase = [], [], []
        proposed_precs_phase_withdups = [] # true representation of model predictions, for calc_accs() 
        prod_smiles_mapped_phase = [] # helper for analyse_proposed() 
        phase_topk = args.topk if phase == 'train' else args.maxk
        dup_count = 0 # count dup proposals per product

        rxn_idx = 0 # we assume order of products in raw_preds is exactly the same as that in clean_rxnsmi_phase, which is true
        is_prod = True # whether current line is the product, or if False --> current line is proposal
        with without_rdkit_log(): # disable RDKit logs because too many messages, eg unable to Kekulize etc
            for line_idx in tqdm(range(len(raw_preds)), desc=f'Reading preds text file of phase {phase} from MEGAN'):
                line_split = raw_preds[line_idx].split(' ')
                if is_prod:
                    is_prod = False # next line will be proposal, this line is product

                    # process prod_smi
                    prod_smi_map = clean_rxnsmi_phase[rxn_idx].split('>>')[-1]
                    prod_mol = Chem.MolFromSmiles(prod_smi_map)
                    [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
                    prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
                    # Sometimes stereochem takes another canonicalization... (just in case)
                    prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)
                    prod_smiles_phase.append(prod_smi_nomap)
                    prod_smiles_mapped_phase.append(prod_smi_map)

                    # process true rcts_smi
                    rcts_smi_map = clean_rxnsmi_phase[rxn_idx].split('>>')[0]
                    rcts_mol = Chem.MolFromSmiles(rcts_smi_map)
                    [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
                    rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
                    # Sometimes stereochem takes another canonicalization...
                    rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)
                    rcts_smiles_phase.append(rcts_smi_nomap)

                    # assert rcts_smi_nomap == line_split[2] # not sure if this will always be true unless really fully canonicalized

                    # reset this_rxn_precursor and this_rxn_seen list
                    this_rxn_precursors, this_rxn_seen = [], [] # seen is de-duplicated, precursors may have duplicates

                elif line_split[0] == '\n':
                    is_prod = True # next line will be product, we finished proposals for previous product

                    proposals_phase[prod_smi_map] = this_rxn_precursors
                    proposed_precs_phase_withdups.append(this_rxn_precursors)

                    if len(this_rxn_seen) < phase_topk: # pad
                        this_rxn_seen.extend(['9999'] * (phase_topk - len(this_rxn_seen)))
                    else: # trim, if needed
                        this_rxn_seen = this_rxn_seen[:phase_topk]
                    proposed_precs_phase.append(this_rxn_seen)
                    
                    rxn_idx += 1

                else: # this line is proposal, directly retrieve unmapped proposal
                    proposal_nomap = line_split[2]
                    # canonicalize, just in case (try to)
                    try:
                        proposal_nomap_mol = Chem.MolFromSmiles(proposal_nomap)
                        Chem.SanitizeMol(proposal_nomap_mol)
                        
                        if proposal_nomap_mol is not None:
                            proposal_nomap = Chem.MolToSmiles(proposal_nomap_mol, True)
                            
                            this_rxn_precursors.append(proposal_nomap)
                            if proposal_nomap not in this_rxn_seen:
                                this_rxn_seen.append(proposal_nomap)
                            else:
                                dup_count += 1
                    except: # unparsable? could be errors like explicit valence greater than permitted, or can't kekulize mol, we can't use this proposal, skip it
                        continue
                    
                    # to get mapped first then clear atom map, but redundant since we can get unmapped directly
                    # proposal_map = line_split[1]
                    # proposal_mol = Chem.MolFromSmiles(proposal_map)
                    # [atom.ClearProp('molAtomMapNumber') for atom in proposal_mol.GetAtoms()]
                    # proposal_nomap = Chem.MolToSmiles(proposal_mol, True)
                    # # Sometimes stereochem takes another canonicalization...
                    # proposal_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(proposal_nomap), True)

        dup_count /= len(clean_rxnsmi_phase)
        logging.info(f'Avg # dups per product: {dup_count}')

        # match predictions to true_precursors & get rank
        logging.info('\nCalculating ranks before removing duplicates')
        _ = calc_accs( 
            [phase],
            clean_rxnsmi_phase,
            rcts_smiles_phase,
            proposed_precs_phase_withdups,
        ) # just to calculate accuracy

        logging.info('\nCalculating ranks after removing duplicates')
        ranks_dict = calc_accs(
                    [phase],
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    proposed_precs_phase
                ) # note that for training data, calc_accs removes the GT proposal if it exists, as we separately append the GT later for all training rxns
        ranks_phase = ranks_dict[phase]
        if phase == 'train':
            logging.info('\n(For training only) Double checking accuracy after removing ground truth predictions, should be all 0%')
            _ = calc_accs(
                    [phase],
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    proposed_precs_phase
                )
        
        analyse_proposed(
            prod_smiles_phase,
            prod_smiles_mapped_phase,
            proposals_phase, # this func needs this to be a dict {mapped_prod_smi: proposals}
        )
        
        combined = {} 
        zipped = []
        for rxn_smi, prod_smi, rcts_smi, rank_of_true_precursor, proposed_rcts_smi in zip(
            clean_rxnsmi_phase,
            prod_smiles_phase,
            rcts_smiles_phase,
            ranks_phase,
            proposed_precs_phase,
        ):
            result = []
            result.extend([rxn_smi, prod_smi, rcts_smi, rank_of_true_precursor])
            result.extend(proposed_rcts_smi)
            zipped.append(result)

        combined[phase] = zipped
        logging.info('Zipped all info for each rxn_smi into a list for dataframe creation!')

        temp_dataframe = pd.DataFrame(
            data={'zipped': combined[phase]}
        )
        phase_dataframe = pd.DataFrame(
            temp_dataframe['zipped'].to_list(),
            index=temp_dataframe.index
        )

        if phase == 'train': # true precursor has been removed from the proposals, so whatever is left are negatives
            proposed_col_names = [f'neg_precursor_{i}' for i in range(1, args.topk + 1)]
        else: # validation/testing, we don't assume true precursor is present & we also do not remove them if present
            proposed_col_names = [f'cand_precursor_{i}' for i in range(1, args.maxk + 1)]
        col_names = ['orig_rxn_smi', 'prod_smi', 'true_precursors', 'rank_of_true_precursor']
        col_names.extend(proposed_col_names)
        phase_dataframe.columns = col_names

        phase_dataframe.to_csv(
            DATA_FOLDER / 
            f'megan_{args.topk}topk_{args.maxk}maxk_noGT_{args.seed}_{phase}.csv',
            index=False
        )
        logging.info(f'Saved proposals of {phase} as CSV!')

def calc_accs( 
            phases : List[str],
            clean_rxnsmi_phase : List[str],
            rcts_smiles_phase : List[str],
            proposed_precs_phase : List[str],
            ) -> Dict[str, List[int]]:
    # boilerplate code
    ranks = {}
    for phase in phases:
        phase_ranks = []
        if phase == 'train':
            for idx in tqdm(range(len(clean_rxnsmi_phase))):
                true_precursors = rcts_smiles_phase[idx]
                all_proposed_precursors = proposed_precs_phase[idx]

                found = False
                for rank, proposal in enumerate(all_proposed_precursors): # ranks are 0-indexed 
                    if true_precursors == proposal:
                        phase_ranks.append(rank)
                        # remove true precursor from proposals 
                        all_proposed_precursors.pop(rank) 
                        all_proposed_precursors.append('9999')
                        found = True
                        break

                if not found:
                    phase_ranks.append(9999)    
        else:
            for idx in tqdm(range(len(clean_rxnsmi_phase))):
                true_precursors = rcts_smiles_phase[idx]
                all_proposed_precursors = proposed_precs_phase[idx]

                found = False
                for rank, proposal in enumerate(all_proposed_precursors): # ranks are 0-indexed  
                    if true_precursors == proposal:
                        phase_ranks.append(rank) 
                        # do not pop true precursor from proposals! 
                        found = True
                        break

                if not found:
                    phase_ranks.append(9999) 
        ranks[phase] = phase_ranks

        logging.info('\n')
        for n in [1, 3, 5, 10, 20, 50, 100, 200]:
            total = float(len(ranks[phase]))
            acc = sum([r+1 <= n for r in ranks[phase]]) / total
            logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
        logging.info('\n')

    return ranks # dictionary 

def analyse_proposed(
                    prod_smiles_phase : List[str],
                    prod_smiles_mapped_phase : List[str],
                    proposals_phase : Dict[str, List[str]],
                    ):
    # boilerplate code                    
    proposed_counter = Counter()
    total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
    key_count = 0
    for key, mapped_key in zip(prod_smiles_phase, prod_smiles_mapped_phase): 
        precursors = proposals_phase[mapped_key]
        precursors_count = len(precursors)
        total_proposed += precursors_count
        if precursors_count > max_proposed:
            max_proposed = precursors_count
            prod_smi_max = key
        if precursors_count < min_proposed:
            min_proposed = precursors_count
            prod_smi_min = key
        
        proposed_counter[key] = precursors_count
        key_count += 1
        
    logging.info(f'Average precursors proposed per prod_smi (dups removed): {total_proposed / key_count}')
    logging.info(f'Min precursors: {min_proposed} for {prod_smi_min}')
    logging.info(f'Max precursors: {max_proposed} for {prod_smi_max})')

    logging.info(f'\nMost common 20:')
    for i in proposed_counter.most_common(20):
        logging.info(f'{i}')
    logging.info(f'\nLeast common 20:')
    for i in proposed_counter.most_common()[-20:]:
        logging.info(f'{i}')
    return

def parse_args():
    parser = argparse.ArgumentParser("inference.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="compile_megan")
    parser.add_argument("--rxn_smi_prefix", help="rxn_smi file", 
                        type=str, default="50k_clean_rxnsmi_noreagent_allmapped_canon")
    # metadata
    parser.add_argument("--seed", help="Seed used for model training", type=int, default=0)
    parser.add_argument("--phases", help="Phases to do inference on", type=str, 
                        default=['train', 'valid', 'test'], nargs='+')
    parser.add_argument("--topk", help="How many top-k proposals to put in train (not guaranteed)", 
                        type=int, default=50)
    parser.add_argument("--maxk", help="How many top-k proposals to generate and put in valid/test (not guaranteed)", 
                        type=int, default=200)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    os.makedirs("./logs/gen_megan", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(f"./logs/gen_megan/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh) 

    logging.info(f'{args}')
    compile_into_csv(args)