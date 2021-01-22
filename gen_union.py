import csv
import pickle 
import sys
import logging 
import argparse
import os
from collections import Counter
from datetime import datetime

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

from rdkit import RDLogger
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

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

def merge_proposers(
        proposers : List[str],
        proposed_smi_file_prefixes : List[str],
        topks : List[int],
        maxks : List[int],
        input_mode : str = 'csv',
        phases: Optional[List[str]] = ['train', 'valid', 'test'],
        rxnsmi_file_prefix: Optional[str] = '50k_clean_rxnsmi_noreagent_allmapped',
        input_folder: Optional[Union[str, bytes, os.PathLike]] = None,
        output_folder: Optional[Union[str, bytes, os.PathLike]] = None
    ):
    for phase in phases:
        logging.info(f'Processing {phase} of {phases}')

        proposals_phase_dict, topk_dict, maxk_dict = {}, {}, {}
        if input_mode == 'pickle':
            for proposer, file_prefix, topk, maxk in zip(
                proposers, proposed_smi_file_prefixes, topks, maxks
                ):
                if proposer == 'retrosim': # not split into phases
                    with open(input_folder / f'{file_prefix}.pickle', 'rb') as handle:
                        proposals_phase_dict[proposer] = pickle.load(handle)
                else:
                    with open(input_folder / f'{file_prefix}_{phase}.pickle', 'rb') as handle:
                        proposals_phase_dict[proposer] = pickle.load(handle)
                topk_dict[proposer] = topk
                maxk_dict[proposer] = maxk

        elif input_mode == 'csv':
            for proposer, file_prefix, topk, maxk in zip(
                proposers, proposed_smi_file_prefixes, topks, maxks
                ):
                with open(input_folder / f'{file_prefix}_{phase}.csv', 'r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    csv_length = 0
                    for row in csv_reader:
                        csv_length += 1

                with open(input_folder / f'{file_prefix}_{phase}.csv', 'r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    # need to build dictionary: {prod_smi: proposals}
                    proposals_phase_dict[proposer] = {}
                    prefix = 'neg_precursor' if phase == 'train' else 'cand_precursor'
                    phase_topk = topk if phase == 'train' else maxk

                    def cano_prec_helper(row, phase_topk, prefix):
                        rxn_prod_smi = row['prod_smi']
                        rxn_prod_smi = Chem.MolToSmiles(Chem.MolFromSmiles(rxn_prod_smi), True)
                        rxn_precs = []
                        dup_count = 0
                        for i in range(1, phase_topk + 1):
                            prec = row[f'{prefix}_{i}']
                            if str(prec) == '9999':
                                break
                            else:
                                # double canonicalize just in case
                                prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec), True)
                                prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec), True)
                                if prec not in rxn_precs:
                                    rxn_precs.append(prec)
                                else:
                                    dup_count += 1
                        return rxn_prod_smi, rxn_precs, dup_count
                    
                    if len(proposers) > 1: # means union
                        for row in tqdm(csv_reader, desc=f'Reading {proposer} csv', total=csv_length):
                            rxn_prod_smi = row['prod_smi']
                            rxn_prod_smi = Chem.MolToSmiles(Chem.MolFromSmiles(rxn_prod_smi), True)
                            rxn_precs = []
                            dup_count = 0
                            for i in range(1, phase_topk + 1):
                                prec = row[f'{prefix}_{i}']
                                if str(prec) == '9999':
                                    break
                                else: # assume individual CSV files have already been canonicalized
                                    if prec not in rxn_precs:
                                        rxn_precs.append(prec)
                                    else:
                                        dup_count += 1
                            proposals_phase_dict[proposer][rxn_prod_smi] = rxn_precs

                    else: # means just 1, doing cleaning
                        num_cores = len(os.sched_getaffinity(0))
                        logging.info(f'Parallelizing over {num_cores} cores')
                        with tqdm_joblib(tqdm(desc=f'Reading {proposer} csv', total=csv_length)) as progress_bar:
                            results = Parallel(n_jobs=num_cores)(
                                            delayed(cano_prec_helper)(row, phase_topk, prefix) 
                                                for row in csv_reader
                                        )
                            all_rxn_prod_smi, all_rxn_precs, dup_count = zip(*results)
                            all_rxn_prod_smi, all_rxn_precs, dup_count = list(all_rxn_prod_smi), list(all_rxn_precs), list(dup_count)
                            dup_count = sum(dup_count) / csv_length
                            for i, rxn_prod_smi in enumerate(all_rxn_prod_smi):
                                proposals_phase_dict[proposer][rxn_prod_smi] = all_rxn_precs[i]
                    
                topk_dict[proposer] = topk
                maxk_dict[proposer] = maxk
                logging.info(f'Avg # dups per product within {proposer}: {dup_count}')

        with open(input_folder / f'{rxnsmi_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmi_phase = pickle.load(handle)

        phase_topk_dict = topk_dict if phase == 'train' else maxk_dict
        phase_topk_sum = 0
        for proposer, topk in phase_topk_dict.items():
            phase_topk_sum += topk

        proposed_precs_phase, prod_smiles_phase, rcts_smiles_phase = [], [], []
        proposed_precs_phase_withdups = [] # true representation of model predictions, for calc_accs() 
        prod_smiles_mapped_phase = [] # helper for analyse_proposed() 
        dup_count = 0
        for rxn_smi in tqdm(clean_rxnsmi_phase, desc='Processing rxn_smi'):
            # process prod_smi
            prod_smi_map = rxn_smi.split('>>')[-1]
            prod_mol = Chem.MolFromSmiles(prod_smi_map)
            [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
            prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
            # Sometimes stereochem takes another canonicalization... (just in case)
            prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)
            prod_smiles_phase.append(prod_smi_nomap)
            prod_smiles_mapped_phase.append(prod_smi_map)

            # process true rcts_smi
            rcts_smi_map = rxn_smi.split('>>')[0]
            rcts_mol = Chem.MolFromSmiles(rcts_smi_map)
            [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
            rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
            # Sometimes stereochem takes another canonicalization...
            rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)
            rcts_smiles_phase.append(rcts_smi_nomap)

            # process proposals
            precs_this_rxn, precs_this_rxn_withdups = [], []
            if input_mode == 'pickle':
                for proposer in proposers:
                    if proposer == 'GLN_retrain':
                        curr_precs = proposals_phase_dict[proposer][prod_smi_map]['reactants']
                    elif proposer == 'MT':
                        results = proposals_phase_dict[proposer][prod_smi_map]
                        curr_precs = []
                        for pred in results[::-1]: # need to reverse
                            precs, scores = pred
                            precs = '.'.join(precs)
                            curr_precs.append(precs)
                    elif proposer == 'retrosim':
                        curr_precs = proposals_phase_dict[proposer][prod_smi_nomap]

                    # remove duplicate predictions
                    seen = [] # sadly have to use List instd of Set to preserve order (i.e. rank of proposal)
                    for prec in curr_precs:
                        if prec not in seen and prec not in precs_this_rxn:
                            seen.append(prec)
                        else:
                            dup_count += 1 / len(clean_rxnsmi_phase)

                    seen = seen[:phase_topk_dict[proposer]]

                    precs_this_rxn.extend(seen)
                    precs_this_rxn_withdups.extend(curr_precs)

            elif input_mode == 'csv':
                for proposer in proposers:
                    if proposer == 'GLN_retrain' or proposer == 'retrosim' or proposer == 'retroxpert':
                        curr_precs = proposals_phase_dict[proposer][prod_smi_nomap] #[prod_smi_map]['reactants']
                    elif proposer == 'MT':
                        raise NotImplementedError

                    # remove duplicate predictions
                    seen = [] # sadly have to use List instd of Set to preserve order (i.e. rank of proposal)
                    for prec in curr_precs:
                        if prec not in seen and prec not in precs_this_rxn:
                            seen.append(prec)
                        else:
                            dup_count += 1 / len(clean_rxnsmi_phase)

                    seen = seen[:phase_topk_dict[proposer]]

                    precs_this_rxn.extend(seen)
                    precs_this_rxn_withdups.extend(curr_precs)

            if len(precs_this_rxn) < phase_topk_sum:
                precs_this_rxn.extend(['9999'] * (phase_topk_sum - len(precs_this_rxn)))

            proposed_precs_phase.append(precs_this_rxn)
            proposed_precs_phase_withdups.append(precs_this_rxn_withdups)

        logging.info(f'Avg # dups per product: {dup_count}')

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
                )
        ranks_phase = ranks_dict[phase]

        if phase == 'train':
            logging.info('\nDouble checking train accuracy (should be 0%) by this stage')
            _ = calc_accs( 
                    [phase],
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    proposed_precs_phase
                )

        analyse_proposed(
            prod_smiles_phase,
            prod_smiles_mapped_phase,
            proposals_phase_dict,
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
            data={
                'zipped': combined[phase]
            }
        )
        phase_dataframe = pd.DataFrame(
            temp_dataframe['zipped'].to_list(),
            index=temp_dataframe.index
        )

        if phase == 'train': # true precursor has been removed from the proposals, so whatever is left are negatives
            proposed_col_names = [f'neg_precursor_{i}' for i in range(1, phase_topk_sum + 1)]
        else: # validation/testing, we don't assume true precursor is present & we also do not remove them if present
            proposed_col_names = [f'cand_precursor_{i}' for i in range(1, phase_topk_sum + 1)]
        base_col_names = ['orig_rxn_smi', 'prod_smi', 'true_precursors', 'rank_of_true_precursor']
        base_col_names.extend(proposed_col_names)
        phase_dataframe.columns = base_col_names

        output_prefix = ''
        for proposer, topk, maxk in zip(
            proposers, topks, maxks
        ):
            output_prefix += f'{proposer}_{topk}topk_{maxk}maxk_'
        phase_dataframe.to_csv(
            output_folder / f'{output_prefix}noGT_{phase}.csv',
            index=False
        )

    logging.info(f'Saved proposals as a dataframe in {output_folder}!')
    return

def calc_accs( 
        phases : List[str],
        clean_rxnsmi_phase : List[str],
        rcts_smiles_phase : List[str],
        proposed_precs_phase : List[str],
    ) -> Dict[str, List[int]]:
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
        for n in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 225, 250, 300]:
            total = float(len(ranks[phase]))
            acc = sum([r+1 <= n for r in ranks[phase]]) / total
            logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
        logging.info('\n')

        # more detailed
        for n in [1] + list(range(5, 301, 5)):
            total = float(len(ranks[phase]))
            acc = sum([r+1 <= n for r in ranks[phase]]) / total
            logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
        logging.info('\n')

    return ranks # dictionary 


def analyse_proposed(
        prod_smiles_phase : List[str],
        prod_smiles_mapped_phase : List[str],
        proposals_phase_dict : Dict[str, # proposer
                                    Dict[str,  # prod_smi
                                        Union[Dict, List] # depends on proposer
                                    ]
                                ],
        input_mode: str = 'csv'
    ): 
    proposed_counter = Counter()
    total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
    key_count = 0
    if input_mode == 'pickle':
        for prod_smi_nomap, prod_smi_map in zip(prod_smiles_phase, prod_smiles_mapped_phase):
            precursors_count = 0
            for proposer, proposals in proposals_phase_dict.items():
                if proposer == 'GLN_retrain':
                    curr_precs = proposals[prod_smi_map]['reactants']  
                    precursors_count += len(curr_precs)
                elif proposer == 'MT':
                    results = proposals[prod_smi_map]
                    curr_precs = []
                    for pred in results[::-1]: # need to reverse
                        precs, scores = pred
                        precs = '.'.join(precs)
                        curr_precs.append(precs)
                    precursors_count += len(curr_precs)
                elif proposer == 'retrosim':
                    curr_precs = proposals[prod_smi_nomap]
                    precursors_count += len(curr_precs)

            total_proposed += precursors_count
            if precursors_count > max_proposed:
                max_proposed = precursors_count
                prod_smi_max = prod_smi_nomap
            if precursors_count < min_proposed:
                min_proposed = precursors_count
                prod_smi_min = prod_smi_nomap

            proposed_counter[prod_smi_nomap] = precursors_count
            key_count += 1
    
    elif input_mode == 'csv':
        for prod_smi_nomap, prod_smi_map in zip(prod_smiles_phase, prod_smiles_mapped_phase):
            precursors_count = 0
            for proposer, proposals in proposals_phase_dict.items():
                if proposer == 'GLN_retrain' or proposer == 'retrosim' or proposer == 'retroxpert':
                    curr_precs = proposals[prod_smi_nomap]
                    precursors_count += len(curr_precs)
                elif proposer == 'MT':
                    raise NotImplementedError

            total_proposed += precursors_count
            if precursors_count > max_proposed:
                max_proposed = precursors_count
                prod_smi_max = prod_smi_nomap
            if precursors_count < min_proposed:
                min_proposed = precursors_count
                prod_smi_min = prod_smi_nomap

            proposed_counter[prod_smi_nomap] = precursors_count
            key_count += 1

    logging.info(f'Average precursors proposed per prod_smi: {total_proposed / key_count}')
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
    parser = argparse.ArgumentParser("gen_union.py")
    parser.add_argument('-f') # filler for COLAB

    parser.add_argument("--log_file", help="log_file", type=str, default="gen_union")
    parser.add_argument("--input_folder", help="input folder", type=str)
    parser.add_argument("--proposed_smi_file_prefixes", 
                        help="List of file prefixes of proposed smiles (comma separated) \
                        in same order as --proposers", 
                        type=str,
                        default="GLN_retrain_cano_200topk_200maxk_200beam,retrosim_200maxtest_200maxprec,retroxpert_200topk_200maxk_200beam")
    parser.add_argument("--input_mode", help="input mode ['csv', 'pickle']", type=str, default='csv')
    parser.add_argument("--rxnsmi_file_prefix", help="file prefix of atom-mapped rxn smiles", type=str,
                        default="50k_clean_rxnsmi_noreagent_allmapped")
    parser.add_argument("--output_folder", help="output folder", type=str)
    parser.add_argument("--location", help="location of script ['COLAB', 'LOCAL']", type=str, default="LOCAL")

    parser.add_argument("--train", help="Whether to compile train preds", action="store_true")
    parser.add_argument("--valid", help="Whether to compile valid preds", action="store_true")
    parser.add_argument("--test", help="Whether to compile test preds", action="store_true")

    parser.add_argument("--proposers", 
                        help="List of proposers (comma separated) \
                        in same order as --proposed_smi_file_prefixes ['GLN_retrain', 'retrosim', 'retroxpert']", 
                        type=str, default='GLN_retrain,retrosim,retroxpert')
    parser.add_argument("--topks", help="List of topk's (comma separated) in same order as --proposers", type=str)
    parser.add_argument("--maxks", help="List of maxk's (comma separated) in same order as --proposers", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args() 

    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/gen_union/", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(f"./logs/gen_union/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.input_folder is None:
        input_folder = Path(__file__).resolve().parents[0] / 'rxnebm/data/cleaned_data/' 
    else:
        input_folder = Path(args.input_folder)
    if args.output_folder is None:
        if args.location == 'COLAB':
            output_folder = Path('/content/gdrive/MyDrive/rxn_ebm/datasets/Retro_Reproduction/Union_proposals/')
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_folder = input_folder
    else:
        output_folder = Path(args.output_folder)

    phases = [] 
    if args.train:
        logging.info('Appending train')
        phases.append('train') 
    if args.valid:
        logging.info('Appending valid')
        phases.append('valid')
    if args.test:
        logging.info('Appending test')
        phases.append('test') 

    logging.info(args)

    merge_proposers(
        proposers=args.proposers.split(','),
        proposed_smi_file_prefixes=args.proposed_smi_file_prefixes.split(','),
        topks=list(map(int, args.topks.split(','))),
        maxks=list(map(int, args.maxks.split(','))),
        phases=phases,
        input_folder=input_folder,
        input_mode=args.input_mode,
        rxnsmi_file_prefix=args.rxnsmi_file_prefix,
        output_folder=output_folder
    ) 