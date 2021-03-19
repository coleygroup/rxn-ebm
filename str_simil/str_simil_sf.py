import logging
import multiprocessing
import os
import itertools
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import selfies as sf
import textdistance
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

# PARAMETERS (now doing permutations), 100k x 5 => ~118 sec total on 32 cores
TRIALS = 1
NUM_STRINGS = 10000 # q fast, 1 x 100k => ~20 sec/rxn on 64 cores on avg (can be much slower or much faster on some rxn)
MAX_INDIV = 300 # max to enumerate, e.g. 1000 = 1000**2 = 1 mil, can take up to 10 min, quite slow, cannot
RXN_COUNT = 1000 # 30 x 100k takes ~10 mins on 64 cores, 10 x 1mil takes ~34 mins on 64 cores
LOG_FREQ = 1000 # log max_simil, best_noncano & invalids every LOG_FREQ num_strings
# CHUNK_SIZE = NUM_STRINGS // num_cores # not sure if this really helps
SEED = 20210307
random.seed(SEED)
# os.environ["PYTHONHASHSEED"] = str(SEED)
# np.random.seed(SEED)
#################

# https://molvs.readthedocs.io/en/latest/api.html#molvs-standardize
def gen_one_smi(task):
    rcts_mol = task # , prod
    found = False
    invalid = 0
    while not found:
        try:
            rcts_noncano = Chem.MolToSmiles(rcts_mol, doRandom=True)
            length = len(rcts_noncano)
            found = True
        except:
            invalid += 1
            # https://github.com/rdkit/rdkit/issues/3264
            # error is encountered in rdkit 2020.09.1.0, & also rdkit 2020.03.6
            continue
    # simil = textdistance.levenshtein.normalized_similarity(rcts_noncano, prod)
    return rcts_noncano,invalid, length #  simil, 

def check_simil(task):
    noncano_joined, prod = task
    simil = textdistance.levenshtein.normalized_similarity(noncano_joined, prod)
    return simil, noncano_joined

def smi_to_sf(smi):
    return sf.encoder(smi)

def simulate_root(): # root + random
    with open('../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_train.pickle', 'rb') as f:
        train = pickle.load(f)
    
    logging.info('#'*70)
    logging.info(f'Simulating ROOT SELFIES: {TRIALS} trials x {RXN_COUNT} rxn_smi x {NUM_STRINGS} random rcts_smi, SEED {SEED}')
    logging.info('#'*70)
    data = {} # key = train_rxn_idx, value = (orig_simil, max_simil, best_rcts_noncano)
            
    try: # may fail on Windows
        num_cores = len(os.sched_getaffinity(0))
    except:
        num_cores = multiprocessing.cpu_count()
    logging.info(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)

    start_all = time.time()
    all_avg_max_simil, all_avg_greedy_simil, all_avg_orig_simil = 0, 0, 0
    all_avg_invalids = 0
    # greedy is basically take orig_smi if its better than any of sampled_smi
    for trial in range(TRIALS):
        avg_max_simil, avg_greedy_simil, avg_orig_simil = 0, 0, 0
        avg_invalid = 0
        start_each = time.time()
        for rxn_idx in random.sample(range(len(train)), RXN_COUNT): # , desc='Processing rxns'):
            rxn = train[rxn_idx]

            rcts_smi = rxn.split('>>')[0]
            rcts_sf = sf.encoder(rcts_smi)
            if len(rcts_smi.split('.')) > 2:
                logging.info(f'Skipping {rxn_idx} for now as it has more than 2 reactants')
                continue

            prod_smi = rxn.split('>>')[-1]
            prod_sf = sf.encoder(prod_smi)
             
            orig_simil = textdistance.levenshtein.normalized_similarity(rcts_sf, prod_sf)

            rct_noncanos_smi = defaultdict(list)
            for rct_idx, rct in enumerate(rcts_smi.split('.')):
                rct_mol = Chem.MolFromSmiles(rct)

                for atom in range(rct_mol.GetNumAtoms()): # enumerate all different roots
                    rct_noncano_smi = Chem.MolToSmiles(rct_mol, rootedAtAtom=atom)
                    # rct_noncano_sf = sf.encoder(rct_noncano_smi)
                    rct_noncanos_smi[rct_idx].append(rct_noncano_smi)
                    # rct_noncanos_sf[rct_idx].append(rct_noncano_sf)

            rct_noncanos_sf = defaultdict(list)
            for rct_idx in rct_noncanos_smi:
                # tasks = []
                # for smi in rct_noncanos_smi[rct_idx]:
                #     tasks.append(smi)
                # for result in tqdm(pool.imap(smi_to_sf, tasks), total=len(tasks),
                #                 desc='Converting SMILES to SELFIES'):
                for smi in rct_noncanos_smi[rct_idx]:
                    rct_noncanos_sf[rct_idx].append(sf.encoder(smi))

            max_simil = 0
            num_rcts = len(rcts_sf.split('.'))
            if num_rcts == 2: # smi_a, smi_b
                tasks = []
                A = set(rct_noncanos_sf[0])
                B = set(rct_noncanos_sf[1])
                for smi_a, smi_b in itertools.product(A, B):
                    noncano_joined = '.'.join([smi_a, smi_b])
                    tasks.append((noncano_joined, prod_sf))
                for simil, noncano_joined in tqdm(pool.imap(check_simil, tasks), 
                    total=len(tasks), desc='Checking all permutations (2 rcts)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = noncano_joined
            elif num_rcts == 1: # only one rct
                tasks = [(smi, prod_sf) for smi in set(rct_noncanos_sf[0])]
                for simil, smi in tqdm(pool.imap(check_simil, tasks),
                    total=len(tasks), desc='Checking all permutations (1 rct)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = smi
            else: # for num_rcts = 3 or more, we just do itertools.product(A, B, C) and so on, but time complexity is too high
                raise NotImplementedError

            data[rxn_idx] = {
                'orig_simil': orig_simil,
                'max_simil': max_simil,
                'best_noncano': best_noncano,
                # 'max_simils_list': max_simils, 
                # 'best_noncanos_list': best_noncanos, 
                # 'lengths': lengths
            }
            avg_max_simil += max_simil / RXN_COUNT
            avg_orig_simil += orig_simil / RXN_COUNT
            avg_greedy_simil += max(max_simil, orig_simil) / RXN_COUNT
            # avg_invalid += invalids / RXN_COUNT

        logging.info(f'\nTrial {trial}, time elapsed: {time.time() - start_each:.3f}')
        logging.info(f'Avg max_simil {avg_max_simil:.6f}')
        logging.info(f'Avg greedy_simil {avg_greedy_simil:.6f}')
        logging.info(f'Avg orig_simil {avg_orig_simil:.6f}')
        # logging.info(f'Avg invalids {avg_invalid:.6f}')
        all_avg_max_simil += avg_max_simil / TRIALS
        all_avg_orig_simil += avg_orig_simil / TRIALS
        all_avg_greedy_simil += avg_greedy_simil / TRIALS
        # all_avg_invalids += avg_invalid / TRIALS

    logging.info(f'\nTrial_avg max_simil {all_avg_max_simil:.6f}')
    logging.info(f'Trial_avg orig_simil {all_avg_orig_simil:.6f}')
    logging.info(f'Trial_avg greedy_simil {all_avg_greedy_simil:.6f}')
    # logging.info(f'Trial_avg invalids {all_avg_invalids:.6f}')
    logging.info(f'% improvement (sampled) {100*(all_avg_max_simil - all_avg_orig_simil) / all_avg_orig_simil:.4f}%')
    logging.info(f'% improvement (greedy) {100*(all_avg_greedy_simil - all_avg_orig_simil) / all_avg_orig_simil:.4f}%')
    logging.info(f'Time elapsed (secs): {time.time() - start_all:.3f}')

    expt_name = f'ROOT_SELFIES_{TRIALS}T_{NUM_STRINGS}N_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
    with open(f'../rxnebm/data/string_similarity/{expt_name}.pickle', 'wb') as f:
        pickle.dump(data, f)

def simulate_root_random(): # root + random
    with open('../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_train.pickle', 'rb') as f:
        train = pickle.load(f)
    
    logging.info('#'*70)
    logging.info(f'Simulating ROOT + RANDOM: {TRIALS} trials x {RXN_COUNT} rxn_smi x {NUM_STRINGS} x MAX {MAX_INDIV} random rcts_smi, SEED {SEED}')
    logging.info('#'*70)
    data = {} # key = train_rxn_idx, value = (orig_simil, max_simil, best_rcts_noncano)
            
    try: # may fail on Windows
        num_cores = len(os.sched_getaffinity(0))
    except:
        num_cores = multiprocessing.cpu_count()
    logging.info(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)

    start_all = time.time()
    all_avg_max_simil, all_avg_greedy_simil, all_avg_orig_simil = 0, 0, 0
    all_avg_invalids = 0
    # greedy is basically take orig_smi if its better than any of sampled_smi
    for trial in range(TRIALS):
        avg_max_simil, avg_greedy_simil, avg_orig_simil = 0, 0, 0
        avg_invalid = 0
        start_each = time.time()
        for rxn_idx in random.sample(range(len(train)), RXN_COUNT): # , desc='Processing rxns'):
            rxn = train[rxn_idx]

            rcts_smi = rxn.split('>>')[0]
            rcts_sf = sf.encoder(rcts_smi)
            if len(rcts_smi.split('.')) > 2:
                logging.info(f'Skipping {rxn_idx} for now as it has more than 2 reactants')
                continue

            prod_smi = rxn.split('>>')[-1]
            prod_sf = sf.encoder(prod_smi)
             
            orig_simil = textdistance.levenshtein.normalized_similarity(rcts_sf, prod_sf)

            rct_noncanos_smi = defaultdict(list)
            for rct_idx, rct in enumerate(rcts_smi.split('.')):
                rct_mol = Chem.MolFromSmiles(rct)

                for atom in range(rct_mol.GetNumAtoms()): # enumerate all different roots
                    rct_noncano_smi = Chem.MolToSmiles(rct_mol, rootedAtAtom=atom)
                    # rct_noncano_sf = sf.encoder(rct_noncano_smi)
                    rct_noncanos_smi[rct_idx].append(rct_noncano_smi)
                    # rct_noncanos_sf[rct_idx].append(rct_noncano_sf)

            invalids = 0
            lengths = []
            for rct_idx, rct in enumerate(rcts_smi.split('.')):
                rct_mol = Chem.MolFromSmiles(rct)
                tasks = [(rct_mol) for _ in range(NUM_STRINGS)]
                
                this_rct_noncanos = set()
                for smi_idx, result in tqdm(enumerate(pool.imap(gen_one_smi, tasks), start=1),
                                            total=len(tasks), desc='Generating non-cano randomly'):
                    rct_noncano, invalid, length = result #  simil,
                    this_rct_noncanos.add(rct_noncano)

                    invalids += invalid / len(tasks)
                    lengths.append(length)
                    
                    if len(this_rct_noncanos) >= MAX_INDIV: # eg NUM_STRINGS may be 10k, but MAX_INDIV maybe 1k
                        rct_noncanos_smi[rct_idx].extend(this_rct_noncanos)
                        break
                    # if smi_idx % LOG_FREQ == 0:
                    #     max_simils.append(max_simil)
                    #     best_noncanos.append(best_noncano)

            rct_noncanos_sf = defaultdict(list)
            for rct_idx in rct_noncanos_smi:
                tasks = []
                for smi in rct_noncanos_smi[rct_idx]:
                    tasks.append(smi)
                for result in tqdm(pool.imap(smi_to_sf, tasks), total=len(tasks),
                                desc='Converting SMILES to SELFIES'):
                    rct_noncanos_sf[rct_idx].append(result)

            max_simil = 0
            num_rcts = len(rcts_sf.split('.'))
            if num_rcts == 2: # smi_a, smi_b
                tasks = []
                A = set(rct_noncanos_sf[0])
                B = set(rct_noncanos_sf[1])
                for smi_a, smi_b in itertools.product(A, B):
                    noncano_joined = '.'.join([smi_a, smi_b])
                    tasks.append((noncano_joined, prod_sf))
                for simil, noncano_joined in tqdm(pool.imap(check_simil, tasks), 
                    total=len(tasks), desc='Checking all permutations (2 rcts)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = noncano_joined
            elif num_rcts == 1: # only one rct
                tasks = [(smi, prod_sf) for smi in set(rct_noncanos_sf[0])]
                for simil, smi in tqdm(pool.imap(check_simil, tasks),
                    total=len(tasks), desc='Checking all permutations (1 rct)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = smi
            else: # for num_rcts = 3 or more, we just do itertools.product(A, B, C) and so on, but time complexity is too high
                raise NotImplementedError

            data[rxn_idx] = {
                'orig_simil': orig_simil,
                'max_simil': max_simil,
                'best_noncano': best_noncano,
                # 'max_simils_list': max_simils, 
                # 'best_noncanos_list': best_noncanos, 
                'lengths': lengths
            }
            avg_max_simil += max_simil / RXN_COUNT
            avg_orig_simil += orig_simil / RXN_COUNT
            avg_greedy_simil += max(max_simil, orig_simil) / RXN_COUNT
            avg_invalid += invalids / RXN_COUNT

        logging.info(f'\nTrial {trial}, time elapsed: {time.time() - start_each:.3f}')
        logging.info(f'Avg max_simil {avg_max_simil:.6f}')
        logging.info(f'Avg greedy_simil {avg_greedy_simil:.6f}')
        logging.info(f'Avg orig_simil {avg_orig_simil:.6f}')
        logging.info(f'Avg invalids {avg_invalid:.6f}')
        all_avg_max_simil += avg_max_simil / TRIALS
        all_avg_orig_simil += avg_orig_simil / TRIALS
        all_avg_greedy_simil += avg_greedy_simil / TRIALS
        all_avg_invalids += avg_invalid / TRIALS

    logging.info(f'\nTrial_avg max_simil {all_avg_max_simil:.6f}')
    logging.info(f'Trial_avg orig_simil {all_avg_orig_simil:.6f}')
    logging.info(f'Trial_avg greedy_simil {all_avg_greedy_simil:.6f}')
    logging.info(f'Trial_avg invalids {all_avg_invalids:.6f}')
    logging.info(f'% improvement (sampled) {100*(all_avg_max_simil - all_avg_orig_simil) / all_avg_orig_simil:.4f}%')
    logging.info(f'% improvement (greedy) {100*(all_avg_greedy_simil - all_avg_orig_simil) / all_avg_orig_simil:.4f}%')
    logging.info(f'Time elapsed (secs): {time.time() - start_all:.3f}')

    expt_name = f'ROOTRANDOM_SELFIES_{TRIALS}T_{NUM_STRINGS}N_{MAX_INDIV}max_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
    with open(f'../rxnebm/data/string_similarity/{expt_name}.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    log_file = f'SELFIES_{TRIALS}T_{NUM_STRINGS}N_{MAX_INDIV}max_{RXN_COUNT}rxn_{SEED}SEED'

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

    simulate_root() # simulate_root_random()
    logging.info('All done!')

