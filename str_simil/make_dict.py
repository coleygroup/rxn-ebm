import argparse
import logging
import multiprocessing
import os
import itertools
import pickle
import random
import re
import sys
import time
from functools import partial
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import textdistance
import selfies as sf
import py_stringmatching as py_sm
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

from token_to_unicode import token_to_unicode_smiles, token_to_unicode_selfies

parser = argparse.ArgumentParser()
parser.add_argument('--expt_name',
                    type=str, help='Custom description of the experiment')
parser.add_argument('--simil_func',
                    type=str, help='["levenshtein", "monge_elkan"]')
parser.add_argument('--simil_type',
                    type=str, help='["string", "token"]', default='string')
parser.add_argument('--language',
                    type=str, help='Language ["smiles", "selfies"]')
parser.add_argument('--max_indiv',
                    type=int, help='Max unique non-cano rct to generate')
parser.add_argument('--num_strings',
                    type=int, help='Max attempts at generating unique non-cano rct')
parser.add_argument('--seed',
                    type=int, help='Random seed', default=20210307)
args = parser.parse_args()

random.seed(args.seed)

if args.simil_func == 'levenshtein':
    simil_func = textdistance.levenshtein.normalized_similarity
else:
    raise ValueError

def tokenize_selfies_from_smiles(smi: str) -> str:
    encoded_selfies = sf.encoder(smi)
    tokens = list(sf.split_selfies(encoded_selfies))
    return tokens

def tokenize_smiles(smi: str) -> str:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def tokenize_and_unicode_selfies(smi: str) -> str:
    encoded_selfies = sf.encoder(smi)
    tokens = list(sf.split_selfies(encoded_selfies))
    translated = ''.join(map(token_to_unicode_selfies.get, tokens))
    return translated

def tokenize_and_unicode_smiles(smi: str) -> str:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    translated = ''.join(map(token_to_unicode_smiles.get, tokens))
    return translated

def check_one_rxn_tokenized_root_SMILES(rxn_smi): # just do root, updated
    rcts_smi = rxn_smi.split('>>')[0]
    prod_smi = rxn_smi.split('>>')[-1]
    
    this_rxn_tokens = set(token_to_unicode_smiles.keys())
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)
        
        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(
                        rct_mol,
                        rootedAtAtom=atom,
                    )
            this_rxn_tokens.update(tokenize_smiles(rct_noncano_smi))

    return this_rxn_tokens

def check_one_rxn_tokenized_SMILES(rxn_smi): # root + random
    # use levenshtein char-level edit dist using unicode dict trick
    # SMILES token vocab size is ~283. ascii is only 255 - 33 - 1 (DEL) = 221
    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_and_unicode_smiles(rcts_smi)

    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_and_unicode_smiles(prod_smi)

    this_rxn_tokens = set(token_to_unicode_smiles.keys())
    rct_noncanos = [] # element = list[noncanos_by_root_for_that_rctidx]
    # len(rct_noncanos) = # of rcts
    rct_noncanos_by_idx_root = {} # key = noncano, value = dict[rct_noncanos_by_root]
    rct_mols_by_idx = {} # key = rct_idx, value = Chem.Mol object
    noncano_to_rct_idx = {} # key = noncano, value = rct_idx (original)
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)
        rct_mols_by_idx[rct_idx] = rct_mol
        
        thisidx_rct_noncanos = []
        rct_noncanos_by_root = {}
        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(
                        rct_mol,
                        rootedAtAtom=atom,
                    )
            rct_noncanos_by_root[rct_noncano_smi] = atom
            thisidx_rct_noncanos.append(rct_noncano_smi)
            noncano_to_rct_idx[rct_noncano_smi] = rct_idx
        rct_noncanos.append(thisidx_rct_noncanos)
        rct_noncanos_by_idx_root[rct_idx] = rct_noncanos_by_root
    
    # enumerate to find best roots for each rct, works for any # of rcts
    max_simil_smi = float('-inf')
    for combo in itertools.product(*rct_noncanos):
        for perm in itertools.permutations(combo):
            noncano_concat = '.'.join(perm)
            noncano_concat_ = tokenize_and_unicode_smiles(noncano_concat)
            simil = simil_func(noncano_concat_, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat
                
    best_roots = {}
    for best_root_rct in best_noncano.split('.'):
        # due to permutation, we lost the original rct_idx, so we need to retrieve it
        rct_idx = noncano_to_rct_idx[best_root_rct]
        
        best_roots[rct_idx] = rct_noncanos_by_idx_root[rct_idx][best_root_rct]
        
        # now we generate noncanos randomly up to specified budget
        # NUM_STRINGS may be 10k, but MAX_INDIV maybe just 100
        counter = 0
        this_rct_noncanos = set([best_root_rct])
        while len(this_rct_noncanos) < args.max_indiv and counter < args.num_strings:
            rct_noncano = Chem.MolToSmiles(
                rct_mols_by_idx[rct_idx], 
                rootedAtAtom=best_roots[rct_idx],
                doRandom=True
            )
            this_rct_noncanos.add(rct_noncano)
            this_rxn_tokens.update(tokenize_smiles(rct_noncano))
            counter += 1

    return this_rxn_tokens

def check_one_rxn_tokenized_root_sf(rxn_smi): # just do root, updated
    rcts_smi = rxn_smi.split('>>')[0]
    prod_smi = rxn_smi.split('>>')[-1]    

    this_rxn_tokens = set(token_to_unicode_selfies.keys())
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)
        
        thisidx_rct_noncanos = []
        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(
                        rct_mol,
                        rootedAtAtom=atom,
                    )
            this_rxn_tokens.update(tokenize_selfies_from_smiles(rct_noncano_smi))

    return this_rxn_tokens

def check_one_rxn_tokenized_sf(rxn_smi):
    # use levenshtein char-level edit dist using unicode dict trick
    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_and_unicode_selfies(rcts_smi)

    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_and_unicode_selfies(prod_smi)
    
    this_rxn_tokens = set(token_to_unicode_selfies.keys())
    rct_noncanos = [] # element = list[noncanos_by_root_for_that_rctidx]
    # len(rct_noncanos) = # of rcts
    rct_noncanos_by_idx_root = {} # key = noncano, value = dict[rct_noncanos_by_root]
    rct_mols_by_idx = {} # key = rct_idx, value = Chem.Mol object
    noncano_to_rct_idx = {} # key = noncano, value = rct_idx (original)
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)
        rct_mols_by_idx[rct_idx] = rct_mol
        
        thisidx_rct_noncanos = []
        rct_noncanos_by_root = {}
        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(
                        rct_mol,
                        rootedAtAtom=atom,
                    )
            rct_noncanos_by_root[rct_noncano_smi] = atom
            thisidx_rct_noncanos.append(rct_noncano_smi)
            noncano_to_rct_idx[rct_noncano_smi] = rct_idx
        rct_noncanos.append(thisidx_rct_noncanos)
        rct_noncanos_by_idx_root[rct_idx] = rct_noncanos_by_root
    
    # enumerate to find best roots for each rct, works for any # of rcts
    max_simil_smi = float('-inf')
    for combo in itertools.product(*rct_noncanos):
        for perm in itertools.permutations(combo):
            noncano_concat = '.'.join(perm)
            noncano_concat_ = tokenize_and_unicode_selfies(noncano_concat)
            simil = simil_func(noncano_concat_, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat
                
    best_roots = {}
    for best_root_rct in best_noncano.split('.'):
        # due to permutation, we lost the original rct_idx, so we need to retrieve it
        rct_idx = noncano_to_rct_idx[best_root_rct]
        
        best_roots[rct_idx] = rct_noncanos_by_idx_root[rct_idx][best_root_rct]
        
        # now we generate noncanos randomly up to specified budget
        # NUM_STRINGS may be 10k, but MAX_INDIV maybe just 100
        counter = 0
        this_rct_noncanos = set([best_root_rct])
        while len(this_rct_noncanos) < args.max_indiv and counter < args.num_strings:
            rct_noncano = Chem.MolToSmiles(
                rct_mols_by_idx[rct_idx], 
                rootedAtAtom=best_roots[rct_idx],
                doRandom=True
            )
            this_rct_noncanos.add(rct_noncano)
            this_rxn_tokens.update(tokenize_selfies_from_smiles(rct_noncano))
            counter += 1

    return this_rxn_tokens

def main():
    try: # may fail on Windows
        num_cores = len(os.sched_getaffinity(0))
    except:
        num_cores = multiprocessing.cpu_count()
    logging.info(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)

    all_vocab = set()
    for phase in ['valid', 'test', 'train']:
        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_{phase}.pickle', 'rb') as f:
            rxns_phase = pickle.load(f)
        
        if args.language == 'selfies':
            if args.simil_type == 'token':
                check_one_rxn_ = check_one_rxn_tokenized_sf
            elif args.simil_type == 'token_rootonly':
                check_one_rxn_ = check_one_rxn_tokenized_root_sf
            else:
                raise ValueError
        elif args.language == 'smiles':
            if args.simil_type == 'token':
                check_one_rxn_ = check_one_rxn_tokenized_SMILES
            elif args.simil_type == 'token_rootonly':
                check_one_rxn_ = check_one_rxn_tokenized_root_SMILES
            else:
                raise ValueError
        else:
            raise ValueError('Unrecognized language')

        for token_vocab in tqdm(pool.imap(check_one_rxn_, rxns_phase),
                        total=len(rxns_phase), desc=f'Processing rxn_smi of {phase}'):
            all_vocab.update(token_vocab) # token_vocab is a set of tokens for that rxn

    if args.language == 'selfies':
        orig_vocab = set(token_to_unicode_selfies.keys())
    elif args.language == 'smiles':
        orig_vocab = set(token_to_unicode_smiles.keys())

    for new_token in all_vocab - orig_vocab:
        print(f'"{new_token}",\n')
    # TODO: save as pickle file instead

if __name__ == "__main__":
    log_file = f'{args.language}_{args.expt_name.upper()}_{args.seed}SEED'

    os.makedirs(Path(__file__).resolve().parents[1] / "logs/str_simil/", exist_ok=True)
    RDLogger.DisableLog('rdApp.*') # to disable all logging from RDKit side
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(Path(__file__).resolve().parents[1] / f"logs/str_simil/{log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logging.info(log_file)
    for phase in ['train', 'valid', 'test']:
        if not Path(
            Path(__file__).resolve().parents[1] / 
            f'rxnebm/data/cleaned_data/50k_rooted_{args.language}_{args.expt_name}_{phase}.pickle'
        ).exists():
            main()
            break

    logging.info('All done!')