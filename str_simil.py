import logging
import multiprocessing
import os
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

try: # may fail on Windows
    num_cores = len(os.sched_getaffinity(0))
except:
    num_cores = multiprocessing.cpu_count()
logging.info(f'Parallelizing over {num_cores} cores')

# PARAMETERS
TRIALS = 1
NUM_STRINGS = 100000 # q fast, 1 x 100k => ~20 sec/rxn on 64 cores
RXN_COUNT = 30 # 30 x 100k takes ~10 mins on 64 cores, 10 x 1mil takes ~34 mins on 64 cores
LOG_FREQ = 1000 # log max_simil, best_noncano & invalids every LOG_FREQ num_strings
# CHUNK_SIZE = NUM_STRINGS // num_cores # not sure if this really helps
SEED = 20210307
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
#################

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
            continue
    simil = textdistance.levenshtein.normalized_similarity(rcts_noncano, prod)
    return rcts_noncano, simil, invalid, length

def simulate():
    with open('rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_canon_train.pickle', 'rb') as f:
        train = pickle.load(f)
    
    logging.info('#'*70)
    logging.info(f'Simulating {TRIALS} trials x {RXN_COUNT} rxn_smi x {NUM_STRINGS} random rcts_smi, SEED {SEED}')
    logging.info('#'*70)
    data = {} # key = train_rxn_idx, value = (orig_simil, max_simil, best_rcts_noncano)

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

            # logging.info(f'\nRxn_rxn_idx {rxn_idx}, time elapsed: {time.time() - start_each:.3f}')
            # logging.info(f'Best sampled: {best_noncano}, simil: {max_simil:.6f}')
            # logging.info(f'Original rct_smi: {rcts}, simil: {orig_simil:.6f}')

            data[rxn_idx] = {
                'orig_simil': orig_simil, 
                'max_simil': max_simil, 
                'best_noncano': best_noncano, 
                'max_simils_list': max_simils, 
                'best_noncanos_list': best_noncanos, 
                'invalids_list': invalids_list
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

    expt_name = f'{TRIALS}T_{NUM_STRINGS}N_{RXN_COUNT}rxn_{SEED}SEED_{avg_max_simil:.3f}_{avg_orig_simil:.3f}'
    with open(f'rxnebm/data/string_similarity/{expt_name}.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    log_file = f'{TRIALS}T_{NUM_STRINGS}N_{RXN_COUNT}rxn_{SEED}SEED'

    RDLogger.DisableLog('rdApp.*') # to disable all logging from RDKit side
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"./logs/str_simil/{log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    simulate()
    logging.info('All done!')
