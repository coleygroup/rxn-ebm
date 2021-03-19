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
import py_stringmatching as py_sm
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

# SMILES + textdistance monge_elkan (ROOTED)
SEED = 20210307
random.seed(SEED)
simil_type = 'string' # 'token'
simil_func = textdistance.levenshtein.normalized_similarity
simil_funcname = 'leven'
# simil_func = textdistance.monge_elkan.normalized_similarity
# simil_funcname = 'monge'
# lv = py_sm.similarity_measure.levenshtein.Levenshtein()
# me = py_sm.similarity_measure.monge_elkan.MongeElkan(sim_func=lv.get_raw_score)

################################################################################################

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

def optimize_one_rxn_string(simil_func, rxn_smi):
    # simil_func = similarity function (monge_elkan)
    prod_smi = rxn_smi.split('>>')[-1]
    # prod_smi_ = tokenize_smiles(prod_smi)

    rcts_smi = rxn_smi.split('>>')[0]
    # rcts_smi_ = tokenize_smiles(rcts_smi)

    orig_simil_smi = simil_func(rcts_smi, prod_smi)
    # orig_simil_smi = simil_func(rcts_smi_, prod_smi_)
    # orig_simil_smi = me.get_raw_score(rcts_smi_, prod_smi_)

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

        for a_, b_ in itertools.product(A, B):
            noncano_concat = a_ + '.' + b_
            simil = simil_func(noncano_concat, prod_smi)
            # simil = me.get_raw_score(noncano_concat, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat

    elif len(rct_noncanos_smi) == 3:
        A = set(rct_noncanos_smi[0])
        B = set(rct_noncanos_smi[1])
        C = set(rct_noncanos_smi[2])

        for a_, b_, c_ in itertools.product(A, B, C):
            noncano_concat = a_ + '.' + b_ + '.' + c_
            simil = simil_func(noncano_concat, prod_smi)
            # simil = me.get_raw_score(noncano_concat, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = noncano_concat

    elif len(rct_noncanos_smi) == 1:
        for smi in set(rct_noncanos_smi[0]):
            simil = simil_func(smi, prod_smi)
            # simil = me.get_raw_score(smi, prod_smi_)
            if simil > max_simil_smi:
                max_simil_smi = simil
                best_noncano = smi
    
    else:
        raise ValueError('We have not accounted for rxn w/ more than 3 reactants!')

    return best_noncano, max_simil_smi, orig_simil_smi

def optimize_one_rxn_tokenized(simil_func, rxn_smi):
    # simil_func = similarity function (monge_elkan)
    prod_smi = rxn_smi.split('>>')[-1]
    prod_smi_ = tokenize_smiles(prod_smi)

    rcts_smi = rxn_smi.split('>>')[0]
    rcts_smi_ = tokenize_smiles(rcts_smi)

    # orig_simil_smi = simil_func(rcts_smi, prod_smi)
    orig_simil_smi = simil_func(rcts_smi_, prod_smi_)
    # orig_simil_smi = me.get_raw_score(rcts_smi_, prod_smi_)

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
            noncano_concat = a_ + ['.'] + b_
            # noncano_concat = "".join(noncano_concat)
            simil = simil_func(noncano_concat, prod_smi_)
            # simil = me.get_raw_score(noncano_concat, prod_smi_)
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
            noncano_concat = a_ + ['.'] + b_ + ['.'] + c_
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

def main():
    try: # may fail on Windows
        num_cores = len(os.sched_getaffinity(0))
    except:
        num_cores = multiprocessing.cpu_count()
    logging.info(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)

    for phase in ['valid', 'test', 'train']:
        with open(f'../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_{phase}.pickle', 'rb') as f:
            rxns_phase = pickle.load(f)
        
        RXN_COUNT = len(rxns_phase)
        
        best_rcts = []
        orig_simil_avg_smi = 0
        greedy_simil_avg = 0

        if simil_type == 'string':
            optimize_one_rxn_ = partial(optimize_one_rxn_string, simil_func)
        else: # 'token'
            optimize_one_rxn_ = partial(optimize_one_rxn_tokenized, simil_func)

        for result in tqdm(pool.imap(optimize_one_rxn_, rxns_phase), # chunksize=5 --> 6 min for valid
                        total=len(rxns_phase), desc=f'Processing rxn_smi of {phase}'):
            best_noncano, max_simil_smi, orig_simil_smi = result

            greedy_simil_avg += max(max_simil_smi, orig_simil_smi) / RXN_COUNT
            orig_simil_avg_smi += orig_simil_smi / RXN_COUNT
            best_rcts.append(best_noncano)
    
        with open(f'../rxnebm/data/cleaned_data/50k_rooted_{simil_funcname}_{phase}.pickle', 'wb') as f:
            pickle.dump(best_rcts, f)
                        
        perc = (greedy_simil_avg - orig_simil_avg_smi) / orig_simil_avg_smi * 100
        logging.info(f'{phase}, {greedy_simil_avg:.3f}, {orig_simil_avg_smi:.3f}, {perc:.3f}%')

def prep_noncanon_mt():
    start = time.time()
    # rxn_class = "UNK"
    for phase in ['train', 'valid', 'test']:
        with open(f'../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_{phase}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)
        with open(f'../rxnebm/data/cleaned_data/50k_rooted_{simil_funcname}_{phase}.pickle', 'rb') as handle:
            rcts_smis = pickle.load(handle)

        with open(f'../rxnebm/data/cleaned_data/retrosynthesis_rooted_{simil_funcname}_{phase}.smi', mode='w') as f:            
            for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                prod_smi = rxn_smi.split('>>')[-1]
                rcts_smi = rcts_smis[i]
                f.write(prod_smi + ' >> ' + rcts_smi + '\n')
            
    print(f'Finished all phases! Elapsed: {time.time() - start:.2f} secs')

def prep_canon_mt():
    start = time.time()
    # rxn_class = "UNK"
    for phase in ['train', 'valid', 'test']:
        with open(f'../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_{phase}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)

        with open(f'../rxnebm/data/cleaned_data/retrosynthesis_canon_{phase}.smi', mode='w') as f:            
            for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                prod_smi = rxn_smi.split('>>')[-1]
                rcts_smi = rxn_smi.split('>>')[0]
                f.write(prod_smi + ' >> ' + rcts_smi + '\n')
            
    print(f'Finished all phases! Elapsed: {time.time() - start:.2f} secs')
    # very fast, ~60 sec for USPTO-50k

if __name__ == "__main__":
    log_file = f'ROOTED_{simil_funcname.upper()}_{SEED}SEED' # {NUM_STRINGS}N_{MAX_INDIV}max_{RXN_COUNT}rxn_

    RDLogger.DisableLog('rdApp.*') # to disable all logging from RDKit side
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"../logs/str_simil/{log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logging.info(log_file)
    for phase in ['train', 'valid', 'test']:
        if not Path(f'../rxnebm/data/cleaned_data/50k_rooted_{simil_funcname}_{phase}.pickle').exists():
            main()
            break

    for phase in ['train', 'valid', 'test']:
        if not Path(f'../rxnebm/data/cleaned_data/retrosynthesis_rooted_{simil_funcname}_{phase}.pickle').exists():
            prep_noncanon_mt()
            break

    for phase in ['train', 'valid', 'test']:
        if not Path(f'../rxnebm/data/cleaned_data/retrosynthesis_canon_{phase}.smi').exists():
            prep_canon_mt()
            break

    logging.info('All done!')