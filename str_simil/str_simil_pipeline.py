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
# TODO: don't use py script to store dictionary, use pickle & automate pipeline from make_dict.py

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

# simil_func = textdistance.monge_elkan.normalized_similarity
# lv = py_sm.similarity_measure.levenshtein.Levenshtein()
# me = py_sm.similarity_measure.monge_elkan.MongeElkan(sim_func=lv.get_raw_score)
##################################################################################

def tokenize_selfies_from_smiles(smi: str) -> str:
    encoded_selfies = sf.encoder(smi)
    tokens = list(sf.split_selfies(encoded_selfies))
    return tokens
#     assert encoded_selfies == "".join(tokens)
#     return " ".join(tokens)

def tokenize_smiles(smi: str) -> str:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
#     assert smi == "".join(tokens)
#     return " ".join(tokens)
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

def optimize_one_rxn_char(simil_func, rxn_smi): # only does root, char-level (eg levenshtein)
    # simil_func = similarity function (monge_elkan)
    prod_smi = rxn_smi.split('>>')[-1]
    rcts_smi = rxn_smi.split('>>')[0]

    orig_simil_smi = simil_func(rcts_smi, prod_smi)
    # orig_simil_smi = simil_func(rcts_smi_, prod_smi_)
    # orig_simil_smi = me.get_raw_score(rcts_smi_, prod_smi_)

    rct_noncanos = [] # element = list[noncanos_by_root_for_that_rctidx]
    # len(rct_noncanos) = # of rcts
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)

        thisidx_rct_noncanos = []
        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(
                        rct_mol,
                        rootedAtAtom=atom,
                    )
            thisidx_rct_noncanos.append(rct_noncano_smi)
        rct_noncanos.append(thisidx_rct_noncanos)
        
    max_simil_smi = float('-inf')
    for combo in itertools.product(*rct_noncanos):
        for perm in itertools.permutations(combo):
            noncano_concat = '.'.join(perm)
            simil = simil_func(noncano_concat, prod_smi)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat

    return best_noncano, max_simil_smi, orig_simil_smi

def optimize_one_rxn_char_v2(simil_func, rxn_smi): # root + random, char-level (eg levenshtein)
    rcts_smi = rxn_smi.split('>>')[0]
    prod_smi = rxn_smi.split('>>')[-1]
    
    orig_simil_smi = simil_func(rcts_smi, prod_smi)
    
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
            simil = simil_func(noncano_concat, prod_smi)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat
                
    best_roots = {}
    gen_noncanos = [] # element = list[noncanos_by_rctidx]
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
            counter += 1
        gen_noncanos.append(this_rct_noncanos)
        
    # now we enumerate to find best combo & perm among these generated (root + random)
    max_simil_smi = float('-inf')
    for combo in itertools.product(*gen_noncanos):
        for perm in itertools.permutations(combo):
            noncano_concat = '.'.join(perm)
            simil = simil_func(noncano_concat, prod_smi)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat

    return best_noncano, max_simil_smi, orig_simil_smi

def optimize_one_rxn_tokenized_SMILES(simil_func, rxn_smi): # root + random
    # use levenshtein char-level edit dist using unicode dict trick
    # SMILES token vocab size is ~283. ascii is only 255 - 33 - 1 (DEL) = 221
    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_and_unicode_smiles(rcts_smi)

    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_and_unicode_smiles(prod_smi)
    
    orig_simil_smi = simil_func(rcts_smi_, prod_smi_)
    
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
    gen_noncanos = [] # element = list[noncanos_by_rctidx]
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
            counter += 1
        gen_noncanos.append(this_rct_noncanos)
        
    # now we enumerate to find best combo & perm among these generated (root + random)
    max_simil_smi = float('-inf')
    for combo in itertools.product(*gen_noncanos):
        for perm in itertools.permutations(combo):
            noncano_concat = '.'.join(perm)
            noncano_concat_ = tokenize_and_unicode_smiles(noncano_concat)
            simil = simil_func(noncano_concat_, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat

    return best_noncano, max_simil_smi, orig_simil_smi

def optimize_one_rxn_tokenized_root_SMILES(simil_func, rxn_smi): # just do root, updated
    # use levenshtein char-level edit dist using unicode dict trick
    # SMILES token vocab size is ~283. ascii is only 255 - 33 - 1 (DEL) = 221
    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_and_unicode_smiles(rcts_smi)

    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_and_unicode_smiles(prod_smi)
    
    orig_simil_smi = simil_func(rcts_smi_, prod_smi_)
    
    rct_noncanos = [] # element = list[noncanos_by_root_for_that_rctidx]
    # len(rct_noncanos) = # of rcts
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)
        
        thisidx_rct_noncanos = []
        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(
                        rct_mol,
                        rootedAtAtom=atom,
                    )
            thisidx_rct_noncanos.append(rct_noncano_smi)
        rct_noncanos.append(thisidx_rct_noncanos)
    
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

    return best_noncano, max_simil_smi, orig_simil_smi


def optimize_one_rxn_tokenized_sf(simil_func, rxn_smi): # root + random
    # use levenshtein char-level edit dist using unicode dict trick
    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_and_unicode_selfies(rcts_smi)

    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_and_unicode_selfies(prod_smi)
    
    orig_simil_smi = simil_func(rcts_smi_, prod_smi_)
    
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
    gen_noncanos = [] # element = list[noncanos_by_rctidx]
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
            counter += 1
        gen_noncanos.append(this_rct_noncanos)
        
    # now we enumerate to find best combo & perm among these generated (root + random)
    max_simil_smi = float('-inf')
    for combo in itertools.product(*gen_noncanos):
        for perm in itertools.permutations(combo):
            noncano_concat = '.'.join(perm)
            noncano_concat_ = tokenize_and_unicode_selfies(noncano_concat)
            simil = simil_func(noncano_concat_, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat

    return best_noncano, max_simil_smi, orig_simil_smi

def optimize_one_rxn_tokenized_root_sf(simil_func, rxn_smi): # just do root, updated
    # use levenshtein char-level edit dist using unicode dict trick
    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_and_unicode_selfies(rcts_smi)

    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_and_unicode_selfies(prod_smi)
    
    orig_simil_smi = simil_func(rcts_smi_, prod_smi_)
    
    rct_noncanos = [] # element = list[noncanos_by_root_for_that_rctidx]
    # len(rct_noncanos) = # of rcts
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)
        
        thisidx_rct_noncanos = []
        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(
                        rct_mol,
                        rootedAtAtom=atom,
                    )
            thisidx_rct_noncanos.append(rct_noncano_smi)
        rct_noncanos.append(thisidx_rct_noncanos)
    
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

    return best_noncano, max_simil_smi, orig_simil_smi

def main():
    try: # may fail on Windows
        num_cores = len(os.sched_getaffinity(0))
    except:
        num_cores = multiprocessing.cpu_count()
    logging.info(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)

    for phase in ['valid', 'test', 'train']:
        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_{phase}.pickle', 'rb') as f:
            rxns_phase = pickle.load(f)
        
        RXN_COUNT = len(rxns_phase)
        
        best_rcts = []
        orig_simil_avg_smi = 0
        greedy_simil_avg = 0

        if args.language == 'selfies':
            if args.simil_type == 'string':
                optimize_one_rxn_ = partial(optimize_one_rxn_char_sf, simil_func)
            elif args.simil_type == 'token':
                optimize_one_rxn_ = partial(optimize_one_rxn_tokenized_sf, simil_func)
            elif args.simil_type == 'token_rootonly':
                optimize_one_rxn_ = partial(optimize_one_rxn_tokenized_root_sf, simil_func)
            else:
                raise ValueError
        elif args.language == 'smiles':
            if args.simil_type == 'string':
                optimize_one_rxn_ = partial(optimize_one_rxn_char_v2, simil_func) # optimize_one_rxn_char
            elif args.simil_type == 'token':
                optimize_one_rxn_ = partial(optimize_one_rxn_tokenized_SMILES, simil_func)
            elif args.simil_type == 'token_rootonly':
                optimize_one_rxn_ = partial(optimize_one_rxn_tokenized_root_SMILES, simil_func)
            else:
                raise ValueError
        else:
            raise ValueError('Unrecognized language')

        for result in tqdm(pool.imap(optimize_one_rxn_, rxns_phase),
                        total=len(rxns_phase), desc=f'Processing rxn_smi of {phase}'):
            best_noncano, max_simil_smi, orig_simil_smi = result

            greedy_simil_avg += max(max_simil_smi, orig_simil_smi) / RXN_COUNT
            orig_simil_avg_smi += orig_simil_smi / RXN_COUNT
            best_rcts.append(best_noncano)
    
        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/50k_rooted_{args.language}_{args.expt_name}_{phase}.pickle', 'wb') as f:
            pickle.dump(best_rcts, f)
                        
        perc = (greedy_simil_avg - orig_simil_avg_smi) / orig_simil_avg_smi * 100
        logging.info(f'{phase}, {greedy_simil_avg:.3f}, {orig_simil_avg_smi:.3f}, {perc:.3f}%')

def prep_noncanon_mt():
    start = time.time()
    for phase in ['train', 'valid', 'test']:
        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_{phase}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)
        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/50k_rooted_{args.language}_{args.expt_name}_{phase}.pickle', 'rb') as handle:
            rcts = pickle.load(handle)

        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/retrosynthesis_rooted_{args.language}_{args.expt_name}_{phase}.smi', mode='w') as f:            
            if args.language == 'smiles':
                # if args.simil_type == 'string': # dont do it here, do it in transformers.py side
                for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                    prod_smi = rxn_smi.split('>>')[-1]
                    rcts_smi = rcts[i]
                    f.write(prod_smi + ' >> ' + rcts_smi + '\n')
                # else:
                #     for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                #         prod_smi = tokenize_smiles(rxn_smi.split('>>')[-1])
                #         rcts_smi = tokenize_smiles(rcts[i])
                #         f.write(prod_smi + ' >> ' + rcts_smi + '\n')
            else:
                for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                    prod_sf = sf.encoder(rxn_smi.split('>>')[-1])
                    rcts_sf = rcts[i]
                    f.write(prod_sf + ' >> ' + rcts_sf + '\n')
            
    print(f'Finished all phases! Elapsed: {time.time() - start:.2f} secs')

def prep_canon_mt():
    start = time.time()
    # rxn_class = "UNK"
    for phase in ['train', 'valid', 'test']:
        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_{phase}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)

        with open(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/retrosynthesis_canon_{phase}.smi', mode='w') as f:            
            for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                prod_smi = rxn_smi.split('>>')[-1]
                rcts_smi = rxn_smi.split('>>')[0]
                f.write(prod_smi + ' >> ' + rcts_smi + '\n')
            
    print(f'Finished all phases! Elapsed: {time.time() - start:.2f} secs')

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

    for phase in ['train', 'valid', 'test']:
        if not Path(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/retrosynthesis_rooted_{args.language}_{args.expt_name}_{phase}.smi').exists():
            prep_noncanon_mt()
            break

    # for phase in ['train', 'valid', 'test']:
    #     if not Path(Path(__file__).resolve().parents[1] / f'rxnebm/data/cleaned_data/retrosynthesis_canon_{phase}.smi').exists():
    #         prep_canon_mt()
    #         break

    logging.info('All done!')



'''
def optimize_one_rxn_char_sf(simil_func, rxn_smi): # OUTDATED, only does root, dont use this
    # simil_func = similarity function (monge_elkan)
    prod_smi = rxn_smi.split('>>')[-1]
    prod_sf = sf.encoder(prod_smi)

    rcts_smi = rxn_smi.split('>>')[0]
    rcts_sf = sf.encoder(rcts_smi)

    orig_simil_sf = simil_func(rcts_sf, prod_sf)

    rct_noncanos_sf = defaultdict(list)
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)

        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(rct_mol, rootedAtAtom=atom)
            rct_noncanos_sf[rct_idx].append(sf.encoder(rct_noncano_smi))

    max_simil_sf = float('-inf')
    if len(rct_noncanos_sf) == 2:
        A = set(rct_noncanos_sf[0])
        B = set(rct_noncanos_sf[1])

        for a_, b_ in itertools.product(A, B):
            for noncano_concat in [
                a_ + '.' + b_, 
                b_ + '.' + a_
            ]:
                simil = simil_func(noncano_concat, prod_sf)
                # simil = me.get_raw_score(noncano_concat, prod_smi_)
                if simil > max_simil_sf:
                    max_simil_sf = simil
                    best_noncano = noncano_concat

    elif len(rct_noncanos_sf) == 3:
        A = set(rct_noncanos_sf[0])
        B = set(rct_noncanos_sf[1])
        C = set(rct_noncanos_sf[2])

        for a_, b_, c_ in itertools.product(A, B, C):
            for noncano_concat in [
                a_ + '.' + b_ + '.' + c_,
                a_ + '.' + c_ + '.' + b_,
                b_ + '.' + c_ + '.' + a_,
                b_ + '.' + a_ + '.' + c_,
                c_ + '.' + b_ + '.' + a_,
                c_ + '.' + a_ + '.' + b_,
            ]:
                simil = simil_func(noncano_concat, prod_sf)
                # simil = me.get_raw_score(noncano_concat, prod_smi_)
                if simil > max_simil_sf:
                    max_simil_sf = simil
                    best_noncano = noncano_concat

    elif len(rct_noncanos_sf) == 1:
        for smi in set(rct_noncanos_sf[0]):
            simil = simil_func(smi, prod_sf)
            # simil = me.get_raw_score(smi, prod_smi_)
            if simil > max_simil_sf:
                max_simil_sf = simil
                best_noncano = smi
    
    else:
        raise ValueError('We have not accounted for rxn w/ more than 3 reactants!')

    return best_noncano, max_simil_sf, orig_simil_sf


def optimize_one_rxn_tokenized_old(simil_func, rxn_smi): # OUTDATED
    # simil_func = similarity function (monge_elkan)
    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_smiles(prod_smi)

    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_smiles(rcts_smi)

    orig_simil_smi = simil_func(rcts_smi_, prod_smi_)

    rct_noncanos_smi = defaultdict(list)
    for rct_idx, rct in enumerate(rcts_smi.split('.')):
        rct_mol = Chem.MolFromSmiles(rct)

        for atom in range(rct_mol.GetNumAtoms()):
            rct_noncano_smi = Chem.MolToSmiles(rct_mol, rootedAtAtom=atom)
            rct_noncanos_smi[rct_idx].append(rct_noncano_smi)

    max_simil_smi = float('-inf')
    if len(rct_noncanos_smi) == 2:
        A = set(rct_noncanos_smi[0])
        B = set(rct_noncanos_smi[1])
        # tokenize everything
        A_, B_ = [], []
        for a in A:
            A_.append(tokenize_smiles(a))
        for b in B:
            B_.append(tokenize_smiles(b))

        for a_, b_ in itertools.product(A_, B_):
            for noncano_concat in [
                a_ + ['.'] + b_, 
                b_ + ['.'] + a_
            ]:
                simil = simil_func(noncano_concat, prod_smi_)
                if simil > max_simil_smi:
                    max_simil_smi = simil
                    best_noncano = noncano_concat

    elif len(rct_noncanos_smi) == 3:
        A = set(rct_noncanos_smi[0])
        B = set(rct_noncanos_smi[1])
        C = set(rct_noncanos_smi[2])
        # tokenize everything
        A_, B_, C_ = [], [], []
        for a in A:
            A_.append(tokenize_smiles(a))
        for b in B:
            B_.append(tokenize_smiles(b))
        for c in C:
            C_.append(tokenize_smiles(c))

        for a_, b_, c_ in itertools.product(A_, B_, C_):
            for noncano_concat in [
                a_ + ['.'] + b_ + ['.'] + c_,
                a_ + ['.'] + c_ + ['.'] + b_,
                b_ + ['.'] + c_ + ['.'] + a_,
                b_ + ['.'] + a_ + ['.'] + c_,
                c_ + ['.'] + b_ + ['.'] + a_,
                c_ + ['.'] + a_ + ['.'] + b_,
            ]:
                simil = simil_func(noncano_concat, prod_smi_)
                # simil = me.get_raw_score(noncano_concat, prod_smi_)
                if simil > max_simil_smi:
                    max_simil_smi = simil
                    best_noncano = noncano_concat

    elif len(rct_noncanos_smi) == 1:
        for smi in set(rct_noncanos_smi[0]):
            smi = tokenize_smiles(smi)
            simil = simil_func(smi, prod_smi_)
            # simil = me.get_raw_score(smi, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = smi
    
    else:
        raise ValueError('We have not accounted for rxn w/ more than 3 reactants!')

    return "".join(best_noncano), max_simil_smi, orig_simil_smi
'''