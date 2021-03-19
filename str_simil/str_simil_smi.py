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
import textdistance
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

# PARAMETERS (now doing permutations), 100k x 5 => ~118 sec total on 32 cores
TRIALS = 1
NUM_STRINGS = 100000 # 10000 # q fast, 1 x 100k => ~20 sec/rxn on 64 cores on avg (can be much slower or much faster on some rxn)
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
    rcts_mol, prod = task
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
    simil = textdistance.levenshtein.normalized_similarity(rcts_noncano, prod)
    return rcts_noncano, simil, invalid, length

def check_simil(task):
    noncano_joined, prod = task
    simil = textdistance.levenshtein.normalized_similarity(noncano_joined, prod)
    return simil, noncano_joined


def simulate_root(): # just root
    with open('../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_train.pickle', 'rb') as f:
        train = pickle.load(f)
    
    logging.info('#'*70)
    logging.info(f'Simulating ROOT: {TRIALS} trials x {RXN_COUNT} rxn_smi x {NUM_STRINGS} random rcts_smi, SEED {SEED}')
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

            rcts = rxn.split('>>')[0]
            if len(rcts.split('.')) > 2:
                logging.info(f'Skipping {rxn_idx} for now as it has more than 2 reactants')
                continue

            prod = rxn.split('>>')[-1]
            orig_simil = textdistance.levenshtein.normalized_similarity(rcts, prod)

            rct_noncanos = defaultdict(list)
            for rct_idx, rct in enumerate(rcts.split('.')):
                rct_mol = Chem.MolFromSmiles(rct)

                for atom in range(rct_mol.GetNumAtoms()): # enumerate all different roots
                    rct_noncano = Chem.MolToSmiles(rct_mol, rootedAtAtom=atom)
                    rct_noncanos[rct_idx].append(rct_noncano)

            max_simil = 0
            num_rcts = len(rcts.split('.'))
            if num_rcts == 2: # smi_a, smi_b
                tasks = []
                A = set(rct_noncanos[0])
                B = set(rct_noncanos[1])
                for smi_a, smi_b in itertools.product(A, B):
                    noncano_joined = '.'.join([smi_a, smi_b])
                    tasks.append((noncano_joined, prod))
                for simil, noncano_joined in tqdm(pool.imap(check_simil, tasks), 
                    total=len(tasks), desc='Checking all permutations (2 rcts)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = noncano_joined
            elif num_rcts == 1: # only one rct
                tasks = [(smi, prod) for smi in set(rct_noncanos[0])]
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

    expt_name = f'ROOT_{TRIALS}T_{NUM_STRINGS}N_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
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

            rcts = rxn.split('>>')[0]
            if len(rcts.split('.')) > 2:
                logging.info(f'Skipping {rxn_idx} for now as it has more than 2 reactants')
                continue

            prod = rxn.split('>>')[-1]
            orig_simil = textdistance.levenshtein.normalized_similarity(rcts, prod)

            rct_noncanos = defaultdict(list)
            for rct_idx, rct in enumerate(rcts.split('.')):
                rct_mol = Chem.MolFromSmiles(rct)

                for atom in range(rct_mol.GetNumAtoms()): # enumerate all different roots
                    rct_noncano = Chem.MolToSmiles(rct_mol, rootedAtAtom=atom)
                    rct_noncanos[rct_idx].append(rct_noncano)
            
            invalids = 0
            lengths = []
            for rct_idx, rct in enumerate(rcts.split('.')):
                rct_mol = Chem.MolFromSmiles(rct)
                tasks = [(rct_mol, prod) for _ in range(NUM_STRINGS)]
                
                this_rct_noncanos = set()
                for smi_idx, result in tqdm(enumerate(pool.imap(gen_one_smi, tasks), start=1),
                                            total=len(tasks), desc='Generating non-cano randomly'):
                    rct_noncano, simil, invalid, length = result
                    this_rct_noncanos.add(rct_noncano)

                    invalids += invalid / len(tasks)
                    lengths.append(length)
                    
                    if len(this_rct_noncanos) >= MAX_INDIV: # eg NUM_STRINGS may be 10k, but MAX_INDIV maybe 1k
                        rct_noncanos[rct_idx].extend(this_rct_noncanos)
                        break
                    # if smi_idx % LOG_FREQ == 0:
                    #     max_simils.append(max_simil)
                    #     best_noncanos.append(best_noncano)

            max_simil = 0
            num_rcts = len(rcts.split('.'))
            if num_rcts == 2: # smi_a, smi_b
                tasks = []
                A = set(rct_noncanos[0])
                B = set(rct_noncanos[1])
                for smi_a, smi_b in itertools.product(A, B):
                    noncano_joined = '.'.join([smi_a, smi_b])
                    tasks.append((noncano_joined, prod))
                for simil, noncano_joined in tqdm(pool.imap(check_simil, tasks), 
                    total=len(tasks), desc='Checking all permutations (2 rcts)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = noncano_joined
            elif num_rcts == 1: # only one rct
                tasks = [(smi, prod) for smi in set(rct_noncanos[0])]
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

    expt_name = f'ROOTRANDOM_{TRIALS}T_{NUM_STRINGS}N_{MAX_INDIV}max_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
    with open(f'../rxnebm/data/string_similarity/{expt_name}.pickle', 'wb') as f:
        pickle.dump(data, f)


def simulate_whole():
    with open('../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_train.pickle', 'rb') as f:
        train = pickle.load(f)
    
    logging.info('#'*70)
    logging.info(f'Simulating WHOLE: {TRIALS} trials x {RXN_COUNT} rxn_smi x {NUM_STRINGS} random rcts_smi, SEED {SEED}')
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
        for rxn_idx in tqdm(random.sample(range(len(train)), RXN_COUNT)):
            rxn = train[rxn_idx]

            rcts = rxn.split('>>')[0]
            rcts_mol = Chem.MolFromSmiles(rcts)
            prod = rxn.split('>>')[1]
            orig_simil = textdistance.levenshtein.normalized_similarity(rcts, prod)
            tasks = [(rcts_mol, prod) for _ in range(NUM_STRINGS)]
            
            max_simil = 0
            best_noncano = ''
            invalids = 0
            max_simils = []
            best_noncanos = []
            invalids_list = []
            lengths = []
            for smi_idx, result in tqdm(enumerate(pool.imap(gen_one_smi, tasks), start=1),
                                        total=NUM_STRINGS): # , chunksize=CHUNK_SIZE
                rcts_noncano, simil, invalid, length = result
                if simil > max_simil:
                    max_simil = simil
                    best_noncano = rcts_noncano
                if smi_idx % LOG_FREQ == 0:
                    max_simils.append(max_simil)
                    best_noncanos.append(best_noncano)
                    invalids_list.append(invalid)
                invalids += invalid / NUM_STRINGS
                lengths.append(length)

            # logging.info(f'\nRxn_rxn_idx {rxn_idx}, time elapsed: {time.time() - start_each:.3f}')
            # logging.info(f'Best sampled: {best_noncano}, simil: {max_simil:.6f}')
            # logging.info(f'Original rct_smi: {rcts}, simil: {orig_simil:.6f}')

            data[rxn_idx] = {
                'orig_simil': orig_simil, 
                'max_simil': max_simil, 
                'best_noncano': best_noncano, 
                'max_simils_list': max_simils, 
                'best_noncanos_list': best_noncanos, 
                'invalids_list': invalids_list,
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

    expt_name = f'WHOLE_{TRIALS}T_{NUM_STRINGS}N_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
    with open(f'../rxnebm/data/string_similarity/{expt_name}.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    log_file = f'{TRIALS}T_{NUM_STRINGS}N_{MAX_INDIV}max_{RXN_COUNT}rxn_{SEED}SEED'

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

    simulate_root() # simulate_whole() # simulate_root_random()
    logging.info('All done!')


def simulate_indiv():
    with open('../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_train.pickle', 'rb') as f:
        train = pickle.load(f)
    
    logging.info('#'*70)
    logging.info(f'Simulating {TRIALS} trials x {RXN_COUNT} rxn_smi x {NUM_STRINGS} random rcts_smi, SEED {SEED}')
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

            rcts = rxn.split('>>')[0]
            if len(rcts.split('.')) > 2:
                logging.info(f'Skipping {rxn_idx} for now as it has more than 2 reactants')
                continue

            prod = rxn.split('>>')[1]
            tasks = []
            for rct in rcts.split('.'):
                rct_mol = Chem.MolFromSmiles(rct)
                tasks.extend([(rct_mol, prod) for _ in range(NUM_STRINGS)])

            orig_simil = textdistance.levenshtein.normalized_similarity(rcts, prod)
            invalids = 0
            lengths = []
            rcts_noncanos = [] # defaultdict(list)
            rct_idx = 0
            # max_simils = []
            # best_noncanos = []
            for smi_idx, result in tqdm(enumerate(pool.imap(gen_one_smi, tasks), start=1),
                                        total=len(tasks), desc='Generating non-cano randomly'):
                rct_noncano, simil, invalid, length = result
                rcts_noncanos.append(rct_noncano)
                # doesnt seem to work for rxn with 2 rcts :(
                # rcts_noncanos[rct_idx].append(rct_noncano)
                # if len(rcts_noncanos[rct_idx]) >= NUM_STRINGS * (rct_idx + 1):
                #     rct_idx += 1
                invalids += invalid / len(tasks)
                lengths.append(length)
                # if smi_idx % LOG_FREQ == 0:
                #     max_simils.append(max_simil)
                #     best_noncanos.append(best_noncano)

            max_simil = 0
            rct_idx = len(rcts_noncanos) / NUM_STRINGS
            if rct_idx == 2: # smi_a, smi_b
                tasks = []
                A = set(rcts_noncanos[:NUM_STRINGS])
                B = set(rcts_noncanos[NUM_STRINGS:])
                for smi_a, smi_b in itertools.product(A, B):
                    noncano_joined = '.'.join([smi_a, smi_b])
                    tasks.append((noncano_joined, prod))
                for simil, noncano_joined in tqdm(pool.imap(check_simil, tasks), 
                    total=len(tasks), desc='Checking all permutations (2 rcts)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = noncano_joined
            elif rct_idx == 1: # only one rct
                tasks = [(smi, prod) for smi in set(rcts_noncanos)]
                for simil, smi in tqdm(pool.imap(check_simil, tasks),
                    total=len(tasks), desc='Checking all permutations (1 rct)'):
                    if simil > max_simil:
                        max_simil = simil
                        best_noncano = smi
            else: # for rct_idx = 3 or more, we just do itertools.product(A, B, C) and so on, but time complexity is too high
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

    expt_name = f'INDIV_{TRIALS}T_{NUM_STRINGS}N_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
    with open(f'../rxnebm/data/string_similarity/{expt_name}.pickle', 'wb') as f:
        pickle.dump(data, f)

# outdated code 
# (found that optimizing individual rct_smi & finding best permutation is NOT optimal)
# def simulate_perms():
#     with open('../rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_train.pickle', 'rb') as f:
#         train = pickle.load(f)
            
#     try: # may fail on Windows
#         num_cores = len(os.sched_getaffinity(0))
#     except:
#         num_cores = multiprocessing.cpu_count()
#     logging.info(f'Parallelizing over {num_cores} cores')

#     logging.info('#'*70)
#     logging.info(f'Simulating {TRIALS} trials x {RXN_COUNT} rxn_smi x {NUM_STRINGS} random rcts_smi, SEED {SEED}')
#     logging.info('#'*70)
#     data = {} # key = train_rxn_idx, value = (orig_simil, max_simil, best_rcts_noncano)

#     pool = multiprocessing.Pool(num_cores)
#     start_all = time.time()
#     all_avg_max_simil, all_avg_greedy_simil, all_avg_orig_simil = 0, 0, 0
#     all_avg_invalids = 0
#     # greedy is basically take orig_smi if its better than any of sampled_smi
#     for trial in range(TRIALS):
#         avg_max_simil, avg_greedy_simil, avg_orig_simil = 0, 0, 0
#         avg_invalid = 0
#         start_each = time.time()
#         for rxn_idx in tqdm(random.sample(range(len(train)), RXN_COUNT)):
#             rxn = train[rxn_idx]

#             rcts = rxn.split('>>')[0]
#             rcts_list = rcts.split('.')
#             prod = rxn.split('>>')[1]
#             orig_all_simil = textdistance.levenshtein.normalized_similarity(rcts, prod)

#             indiv_bests = []
#             rct_noncanos = {}
#             rct_invalids = {}
#             rct_max_simils = {}
#             rct_to_length = {} # key = individual rct SMILES, value = list of lengths of non-cano rcts SMILES
#             orig_lengths = {} # key = individual rct SMILES, value = length
#             orig_rct_simils = {}
#             total_invalids = 0 # sum across all rct in rcts_list
#             for rct in rcts_list:
#                 rct_mol = Chem.MolFromSmiles(rct)

#                 orig_lengths[rct] = len(rct)
#                 rcts_smi_lengths = [] # store length of smi for each generated non-cano reactant SMILES

#                 orig_rct_simil = textdistance.levenshtein.normalized_similarity(rct, prod)
#                 tasks = [(rct_mol, prod) for _ in range(NUM_STRINGS)]

#                 max_simil = 0
#                 best_noncano = ''
#                 max_simils = []
#                 best_noncanos = []
#                 invalids_list = []
#                 for smi_idx, result in tqdm(enumerate(pool.imap(gen_one_smi, tasks), start=1),
#                                             total=NUM_STRINGS): # , chunksize=CHUNK_SIZE
#                     rct_noncano, simil, invalid, length = result

#                     if simil > max_simil:
#                         max_simil = simil
#                         best_noncano = rct_noncano

#                     if smi_idx % LOG_FREQ == 0:
#                         max_simils.append(max_simil)
#                         best_noncanos.append(best_noncano)
#                         invalids_list.append(invalid)

#                     total_invalids += invalid / NUM_STRINGS
#                     rcts_smi_lengths.append(length)

#                 rct_to_length[rct] = rcts_smi_lengths
#                 rct_noncanos[rct] = best_noncanos
#                 indiv_bests.append(best_noncanos[-1])
#                 rct_invalids[rct] = invalids_list
#                 rct_max_simils[rct] = max_simils
#                 orig_rct_simils[rct] = orig_rct_simil

#             perm_max_simil = 0
#             best_perm = ''
#             for perm in itertools.permutations(indiv_bests):
#                 perm_rcts = '.'.join(perm)
#                 perm_simil = textdistance.levenshtein.normalized_similarity(perm_rcts, prod)
#                 if perm_simil > perm_max_simil:
#                     perm_max_simil = perm_simil
#                     best_perm = perm_rcts

#             data[rxn_idx] = {
#                 'orig_simil': orig_all_simil,
#                 'orig_rct_simils': orig_rct_simils,
#                 'max_simil': perm_max_simil, # max_simil, 
#                 'best_noncano': best_perm, # best_noncano, 
#                 'max_simils_list': rct_max_simils, # max_simils, 
#                 'best_noncanos_list': rct_noncanos, # best_noncanos, 
#                 'invalids_list': rct_invalids, # key: rct_smi, value: invalids_list,
#                 'orig_rxn_smi': rxn,
#             }
#             avg_max_simil += perm_max_simil / RXN_COUNT
#             avg_orig_simil += orig_all_simil / RXN_COUNT
#             avg_greedy_simil += max(perm_max_simil, orig_all_simil) / RXN_COUNT
#             avg_invalid += total_invalids / RXN_COUNT / len(rcts_list)

#             logging.info(f'best: {best_perm} {perm_max_simil}')
#             logging.info(f'orig: {rcts} {orig_all_simil}')

#         logging.info(f'\nTrial {trial}, time elapsed: {time.time() - start_each:.3f}')
#         logging.info(f'Avg max_simil {avg_max_simil:.6f}')
#         logging.info(f'Avg greedy_simil {avg_greedy_simil:.6f}')
#         logging.info(f'Avg orig_simil {avg_orig_simil:.6f}')
#         logging.info(f'Avg invalids {avg_invalid:.6f}')
#         all_avg_max_simil += avg_max_simil / TRIALS
#         all_avg_orig_simil += avg_orig_simil / TRIALS
#         all_avg_greedy_simil += avg_greedy_simil / TRIALS
#         all_avg_invalids += avg_invalid / TRIALS

#     logging.info(f'\nTrial_avg max_simil {all_avg_max_simil:.6f}')
#     logging.info(f'Trial_avg orig_simil {all_avg_orig_simil:.6f}')
#     logging.info(f'Trial_avg greedy_simil {all_avg_greedy_simil:.6f}')
#     logging.info(f'Trial_avg invalids {all_avg_invalids:.6f}')
#     logging.info(f'% improvement (sampled) {100*(all_avg_max_simil - all_avg_orig_simil) / all_avg_orig_simil:.4f}%')
#     logging.info(f'% improvement (greedy) {100*(all_avg_greedy_simil - all_avg_orig_simil) / all_avg_orig_simil:.4f}%')
#     logging.info(f'Time elapsed (secs): {time.time() - start_all:.3f}')

#     expt_name = f'indiv_{TRIALS}T_{NUM_STRINGS}N_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
#     with open(f'../rxnebm/data/string_similarity/{expt_name}.pickle', 'wb') as f:
#         pickle.dump(data, f)
