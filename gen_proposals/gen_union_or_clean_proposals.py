import argparse
import csv
import logging
import multiprocessing
import os
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from joblib import Parallel, delayed
from rdkit import RDLogger
from tqdm import tqdm

from utils import tqdm_joblib

def merge_proposers(
        proposers : List[str],
        proposed_smi_file_prefixes : List[str],
        topks : List[int],
        maxks : List[int],
        phases: Optional[List[str]] = ['train', 'valid', 'test'],
        rxnsmi_file_prefix: Optional[str] = '50k_clean_rxnsmi_noreagent_allmapped_canon',
        input_folder: Optional[Union[str, bytes, os.PathLike]] = None,
        output_file_prefix: Optional[Union[str, bytes, os.PathLike]] = None,
        output_folder: Optional[Union[str, bytes, os.PathLike]] = None,
        seed: Optional[int] = None,
    ):
    for phase in phases:
        logging.info(f'Processing {phase} of {phases}')

        proposals_phase_dict, topk_dict, maxk_dict = {}, {}, {}
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
                        all_rxn_prod_smi, all_rxn_precs, dup_count = map(list, zip(*results))
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
            for proposer in proposers:
                if proposer in ['GLN', 'retrosim', 'retroxpert', 'union']:
                    curr_precs = proposals_phase_dict[proposer][prod_smi_nomap]
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
        _, _ = calc_accs( 
            phase,
            clean_rxnsmi_phase,
            rcts_smiles_phase,
            proposed_precs_phase_withdups,
        ) # just to calculate accuracy

        logging.info('\nCalculating ranks after removing duplicates')
        ranks_phase, processed_precs_phase = calc_accs( 
                    phase,
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    proposed_precs_phase,
                    maxk
                )

        if phase == 'train':
            logging.info('\nDouble checking train accuracy (should be 0%) by this stage')
            _, _ = calc_accs( 
                    phase,
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    processed_precs_phase, # proposed_precs_phase
                    maxk
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
            processed_precs_phase, 
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

        if output_file_prefix is None:
            output_file_prefix = ''
            for proposer, topk, maxk in zip(
                proposers, topks, maxks
            ):
                output_file_prefix += f'{proposer}_{topk}topk_{maxk}maxk_'
            if seed:
                phase_dataframe.to_csv(
                    output_folder / f'{output_file_prefix}noGT_{seed}_{phase}.csv',
                    index=False
                )
            else:
                phase_dataframe.to_csv(
                    output_folder / f'{output_file_prefix}noGT_{phase}.csv',
                    index=False
                )
        else: # user provided output_file_prefix, means they know exactly what they want
            logging.info(f'Saving to output_file_prefix {output_file_prefix}')
            phase_dataframe.to_csv(
                    output_folder / f'{output_file_prefix}_{phase}.csv',
                    index=False
                )
    logging.info(f'Saved proposals as a dataframe in {output_folder}!')
    return

def calc_accs( 
        phase : str,
        clean_rxnsmi_phase : List[str],
        rcts_smiles_phase : List[str],
        proposed_precs_phase : List[str],
        maxk : int = 400,
    ) -> Dict[str, List[int]]:
    phase_ranks = []
    processed_precs = []
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

            processed_precs.append(all_proposed_precursors)
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

            processed_precs.append(all_proposed_precursors)

    logging.info('\n')
    for n in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 225, 250, 300, maxk]:
        total = float(len(phase_ranks))
        acc = sum([r+1 <= n for r in phase_ranks]) / total
        logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
    logging.info('\n')

    # more detailed, for debugging
    for n in [1] + list(range(5, 301, 5)) + [maxk]:
        total = float(len(phase_ranks))
        acc = sum([r+1 <= n for r in phase_ranks]) / total
        logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
    logging.info('\n')

    return phase_ranks, processed_precs


def analyse_proposed(
        prod_smiles_phase : List[str],
        prod_smiles_mapped_phase : List[str],
        proposals_phase_dict : Dict[str, # proposer
                                    Dict[str,  # prod_smi
                                        Union[Dict, List] # depends on proposer
                                    ]
                                ],
    ): 
    proposed_counter = Counter()
    total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
    key_count = 0
    for prod_smi_nomap, prod_smi_map in zip(prod_smiles_phase, prod_smiles_mapped_phase):
        precursors_count = 0
        for proposer, proposals in proposals_phase_dict.items():
            if proposer == 'GLN' or proposer == 'retrosim' or proposer == 'retroxpert':
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
    parser.add_argument("--seed", help="seed used for trained model", type=int)
    parser.add_argument("--proposed_smi_file_prefixes", 
                        help="List of input file prefixes of proposed smiles (comma separated) \
                        in same order as --proposers", 
                        type=str,
                        default="GLN_200topk_200maxk_noGT_19260817,retrosim_200topk_200maxk_noGT,retroxpert_200topk_200maxk_noGT_0")
    parser.add_argument("--rxnsmi_file_prefix", help="file prefix of atom-mapped rxn smiles", type=str,
                        default="50k_clean_rxnsmi_noreagent_allmapped_canon")
    parser.add_argument("--output_folder", help="output folder", type=str)
    parser.add_argument("--output_file_prefix", help="output file prefix", type=str)
    parser.add_argument("--phases", help="Phases to generate proposals or compile proposals for, default ['train', 'valid', 'test']", 
                        type=str, nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument("--proposers", 
                        help="List of proposers (comma separated) \
                        in same order as --proposed_smi_file_prefixes ['GLN', 'retrosim', 'retroxpert', 'neuralsym']", 
                        type=str, default='GLN,retrosim,retroxpert')
    parser.add_argument("--topks", help="List of topk's (comma separated) in same order as --proposers", type=str)
    parser.add_argument("--maxks", help="List of maxk's (comma separated) in same order as --proposers", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args() 

    RDLogger.DisableLog("rdApp.warning")

    os.makedirs(Path(__file__).resolve().parents[1] / "logs/gen_union/", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(Path(__file__).resolve().parents[1] / f"logs/gen_union/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.input_folder is None:
        input_folder = Path(__file__).resolve().parents[1] / 'rxnebm/data/cleaned_data/' 
    else:
        input_folder = Path(args.input_folder)
    if args.output_folder is None:
        output_folder = input_folder
    else:
        output_folder = Path(args.output_folder)

    logging.info(args)

    merge_proposers(
        proposers=args.proposers.split(','),
        proposed_smi_file_prefixes=args.proposed_smi_file_prefixes.split(','),
        topks=list(map(int, args.topks.split(','))),
        maxks=list(map(int, args.maxks.split(','))),
        phases=args.phases,
        input_folder=input_folder,
        rxnsmi_file_prefix=args.rxnsmi_file_prefix,
        output_folder=output_folder,
        output_file_prefix=args.output_file_prefix,
        seed=args.seed
    ) 
