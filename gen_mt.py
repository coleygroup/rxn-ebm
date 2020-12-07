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

from rdkit import RDLogger
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from typing import Dict, List

from tensorflow.compat.v1.keras import backend as K

from rxnebm.proposer.mt_karpov_config import mt_karpov_config
from rxnebm.proposer.mt_karpov_proposer import MTKarpovProposer

def merge_chunks(
            topk: int = 50,
            maxk: int = 100,
            beam_size: int = 50,
            temperature: float = 1.3,
            phase: str = 'train',
            start_idxs : List[int] = [0, 13000, 26000],
            end_idxs : List[int] = [13000, 26000, None],
            input_folder: Optional[Union[str, bytes, os.PathLike]] = None,
            input_file_prefix: Optional[str] = '50k_clean_rxnsmi_noreagent_allmapped',
            output_folder: Optional[Union[str, bytes, os.PathLike]] = None
            ):
    """ Helper func to combine separatedly computed chunks into a single chunk, for the specified phase
    """
    merged = {} 
    logging.info(f'Merging start_idxs {start_idxs} and end_idxs {end_idxs}')
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        with open(output_folder / f'MT_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{temperature}T_{phase}_start{start_idx}_end{end_idx}.pickle', 'rb') as handle:
            chunk = pickle.load(handle)
        for key, value in chunk.items(): 
            merged[key] = value
    
    with open(output_folder / f'MT_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{temperature}T_{phase}.pickle', 'wb') as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f'Merged all chunks of {phase}!')
    return 

def gen_proposals(
            topk: int = 50,
            maxk: int = 100, 
            beam_size: int = 50,
            temperature: Optional[float] = 1.3, 
            phases: Optional[List[str]] = ['train', 'valid', 'test'],
            start_idx : Optional[int] = 0,
            end_idx : Optional[int] = None, 
            input_folder: Optional[Union[str, bytes, os.PathLike]] = None,
            input_file_prefix: Optional[str] = '50k_clean_rxnsmi_noreagent_allmapped',
            output_folder: Optional[Union[str, bytes, os.PathLike]] = None,
            checkpoint_every: Optional[int] = 4000,
            ):
    '''
    Parameters
    ----------
    topk : int (Default = 50)
        for each product, how many proposals to put in train
    maxk : int (Default = 100)
        for each product, how many proposals to put in valid/test 
    beam_size : int (Default = 50)  
        beam size for ranking generated proposals
    temperature : float (Default = 1.3)
        temperature for decoding
    phases : List[str] (Default = ['train', 'valid', 'test'])
        phases to generate GLN proposals for
    input_folder : Optional[Union[str, bytes, os.PathLike]] (Default = None)
        path to the folder containing the train/valid/test reaction SMILES strings 
        if None, this defaults to:   path/to/rxn/ebm/data/cleaned_data/ 
    input_file_prefix : Optional[str] (Default = '50k_clean_rxnsmi_noreagent_allmapped')
        prefix of the 3 pickle files containing the train/valid/test reaction SMILES strings
    output_folder : Optional[Union[str, bytes, os.PathLike]] (Default = None)
        path to the folder that will contain the output dicts containing GLN's proposals 
        if None and if location is NOT 'COLAB', this defaults to the same folder as input_data_folder
        otherwise (i.e. we are at 'COLAB'), it defaults to a hardcoded gdrive folder 
    checkpoint_every : Optional[int] (Default = 4000)
        save checkpoint of proposed precursor smiles every N prod_smiles
    ''' 
    proposer = MTKarpovProposer(mt_karpov_config)

    clean_rxnsmis = {} 
    for phase in phases:
        with open(input_folder / f'{input_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmis[phase] = pickle.load(handle)

        phase_proposals = {} # key = prod_smi, value = Dict[template, reactants, scores]
        logging.info(f'Calculting for start_idx: {start_idx}, end_idx: {end_idx}')
        logging.info(f'Using T={temperature}, beam_size={beam_size}, topk={topk}, maxk={maxk}')
        phase_topk = topk if phase == 'train' else maxk
        for i, rxn_smi in enumerate(
                                tqdm(
                                    clean_rxnsmis[phase][ start_idx : end_idx ], 
                                    desc=f'Generating MT Karpov proposals for {phase}'
                                )
                            ):
            prod_smi = rxn_smi.split('>>')[-1]
            prod_mol = Chem.MolFromSmiles(prod_smi)
            [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
            prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)

            rxn_type = ["UNK"]
            try:
                results = proposer.propose([prod_smi_nomap], 
                                            rxn_type, 
                                            topk=phase_topk,
                                            beam_size=beam_size, 
                                            temperature=temperature)
                phase_proposals[prod_smi] = results[0] # results is a list, which itself contains topk lists, each a list [reactants, scores]
            except Exception as e:
                logging.info(f'At index {i} for {prod_smi_nomap}: {e}')
                # put empty list if MT could not propose
                phase_proposals[prod_smi] = []

            if i > 0 and i % checkpoint_every == 0: # checkpoint
                logging.info(f'Checkpointing {i} for {phase}')
                with open(output_folder / 
                        f'MT_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{temperature}T_{phase}_start{start_idx}_end{i + start_idx}.pickle', 
                        'wb') as handle:
                    pickle.dump(phase_proposals, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if start_idx == 0 and end_idx is None:
            with open(output_folder / 
                    f'MT_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{temperature}T_{phase}.pickle', 
                    'wb') as handle:
                pickle.dump(phase_proposals, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_folder / 
                    f'MT_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}_{temperature}T_start{start_idx}_end{end_idx}.pickle', 
                    'wb') as handle:
                pickle.dump(phase_proposals, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f'Successfully finished {phase}!')

    """
    List of n[{"template": List of topk templates,
               "reactants": List of topk reactants,
               "scores": ndarray of topk scores}]
    """
    return

def compile_into_csv(
                topk: int = 50,
                maxk: int = 100,
                beam_size: int = 50,
                temperature: Optional[float] = 1.3,
                phases: Optional[List[str]] = ['train', 'valid', 'test'],
                input_folder: Optional[Union[str, bytes, os.PathLike]] = None,
                input_file_prefix: Optional[str] = '50k_clean_rxnsmi_noreagent_allmapped',
                output_folder: Optional[Union[str, bytes, os.PathLike]] = None
                ):
    for phase in phases:
        logging.info(f'Processing {phase} of {phases}')

        if (output_folder / 
            f'MT_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{temperature}T_{phase}.pickle'
        ).exists(): # file already exists
            with open(output_folder / 
                f'MT_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{temperature}T_{phase}.pickle', 'rb'
            ) as handle:
                proposals_phase = pickle.load(handle) 
        else:
            raise RuntimeError('Error! Could not locate and load GLN proposed smiles file') 
        
        with open(input_folder / f'{input_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmi_phase = pickle.load(handle)
 
        proposed_precs_phase, prod_smiles_phase, rcts_smiles_phase = [], [], []
        prod_smiles_mapped_phase = [] # helper for analyse_proposed() 
        phase_topk = topk if phase == 'train' else maxk
        for rxn_smi in tqdm(clean_rxnsmi_phase, desc='Processing rxn_smi'):     
            prod_smi = rxn_smi.split('>>')[-1]
            prod_mol = Chem.MolFromSmiles(prod_smi)
            [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
            prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
            prod_smiles_phase.append(prod_smi_nomap)
            prod_smiles_mapped_phase.append(prod_smi)
            
            # value for each prod_smi is a dict, where key = 'reactants' retrieves the predicted precursors
            precursors = []
            results = proposals_phase[prod_smi]
            for pred in results:
                this_precs, scores = pred
                this_precs = '.'.join(this_precs)
                precursors.append(this_precs)

            # remove duplicate predictions 
            seen = []
            for prec in precursors:
                if prec not in seen:
                    seen.append(prec)

            if len(seen) < phase_topk:
                seen.extend(['9999'] * (phase_topk - len(seen)))
            else:
                seen = seen[ : phase_topk]
            proposed_precs_phase.append(seen)

            rcts_smi = rxn_smi.split('>>')[0]
            rcts_mol = Chem.MolFromSmiles(rcts_smi)
            [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
            rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
            # Sometimes stereochem takes another canonicalization...
            rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True) 
            rcts_smiles_phase.append(rcts_smi_nomap)

        # logging.info(f'len(precursors): {len(precursors)}')

        ranks_dict = calc_accs( 
            [phase],
            clean_rxnsmi_phase,
            rcts_smiles_phase,
            proposed_precs_phase,
        )
        ranks_phase = ranks_dict[phase]
        analyse_proposed(
            prod_smiles_phase,
            prod_smiles_mapped_phase,
            proposals_phase,
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
        # logging.info('temp_dataframe shape')
        # logging.info(f'{temp_dataframe.shape}')
        
        phase_dataframe = pd.DataFrame(
            temp_dataframe['zipped'].to_list(),
            index=temp_dataframe.index
        ) 
        # logging.info('phase_dataframe shape')
        # logging.info(f'{phase_dataframe.shape}')
        
        if phase == 'train': # true precursor has been removed from the proposals, so whatever is left are negatives
            proposed_col_names = [f'neg_precursor_{i}' for i in range(1, phase_topk + 1)]
        else: # validation/testing, we don't assume true precursor is present & we also do not remove them if present
            proposed_col_names = [f'cand_precursor_{i}' for i in range(1, phase_topk + 1)]
            # logging.info(f'len(proposed_col_names): {len(proposed_col_names)}')
        base_col_names = ['orig_rxn_smi', 'prod_smi', 'true_precursors', 'rank_of_true_precursor']
        base_col_names.extend(proposed_col_names)
        phase_dataframe.columns = base_col_names
        
        # logging.info(f'Shape of {phase} dataframe: {phase_dataframe.shape}')

        phase_dataframe.to_csv(
            output_folder / 
            f'MT_{topk}topk_{maxk}maxk_{beam_size}beam_{temperature}T_{phase}.csv',
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
        for n in [1, 3, 5, 10, 20, 50]:
            total = float(len(ranks[phase]))
            acc = sum([r+1 <= n for r in ranks[phase]]) / total
            logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
        logging.info('\n')

    return ranks # dictionary 

def analyse_proposed(
                    prod_smiles_phase : List[str],
                    prod_smiles_mapped_phase : List[str],
                    proposals_phase : Dict[str, Dict[str, List[str]]],
                    ): 
    proposed_counter = Counter()
    total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
    key_count = 0
    for key, mapped_key in zip(prod_smiles_phase, prod_smiles_mapped_phase): 
        precursors = []
        results = proposals_phase[mapped_key]
        for pred in results:
            this_precs, scores = pred
            this_precs = '.'.join(this_precs)
            precursors.append(this_precs)

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
    parser = argparse.ArgumentParser("gen_mt.py")
    parser.add_argument('-f') # filler for COLAB
    
    parser.add_argument("--log_file", help="log_file", type=str, default="gen_mt")
    parser.add_argument("--input_folder", help="input folder", type=str)
    parser.add_argument("--input_file_prefix", help="input file prefix of atom-mapped rxn smiles", type=str,
                        default="50k_clean_rxnsmi_noreagent_allmapped")
    parser.add_argument("--output_folder", help="output folder", type=str)
    parser.add_argument("--location", help="location of script ['COLAB', 'LOCAL']", type=str, default="COLAB")

    parser.add_argument("--train", help="whether to generate on train data", action="store_true")
    parser.add_argument("--valid", help="whether to generate on valid data", action="store_true")
    parser.add_argument("--test", help="whether to generate on test data", action="store_true")
    parser.add_argument("--start_idx", help="Start idx (train)", type=int, default=0)
    parser.add_argument("--end_idx", help="End idx (train)", type=int)
    parser.add_argument("--checkpoint_every", help="Save checkpoint of proposed smiles every N product smiles",
                        type=int, default=4000)
    
    parser.add_argument("--merge_chunks", help="Whether to merge already computed chunks", action="store_true")
    parser.add_argument("--phase_to_merge", help="Phase to merge chunks of (only supports phase at a time)", type=str)
    parser.add_argument("--compile", help="Whether to compile proposed precursor SMILES (& corresponding rxn_smiles data) into CSV file", 
                        action='store_true')

    parser.add_argument("--beam_size", help="Beam size", type=int, default=50)
    parser.add_argument("--topk", help="How many top-k proposals to put in train (not guaranteed)", type=int, default=50)
    parser.add_argument("--maxk", help="How many top-k proposals to generate and put in valid/test (not guaranteed)", type=int, default=100)
    parser.add_argument("--temperature", help="Temperature for decoding", type=float, default=1.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}")
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
            output_folder = Path('/content/gdrive/MyDrive/rxn_ebm/datasets/Retro_Reproduction/MT_proposals/')
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

    if phases:
        gen_proposals(
            maxk=args.maxk,
            topk=args.topk,
            beam_size=args.beam_size,
            temperature=args.temperature,
            phases=phases,
            start_idx=args.start_idx,
            end_idx=args.end_idx, 
            input_folder=input_folder,
            input_file_prefix=args.input_file_prefix,
            output_folder=output_folder,
            checkpoint_every=args.checkpoint_every
        ) 
    
    if args.merge_chunks:
        merge_chunks(
            topk=args.topk,
            beam_size=args.beam_size,
            temperature=args.temperature,
            phase=args.phase_to_merge,
            start_idxs=[0, 13000, 26000],
            end_idxs=[13000, 26000, None],
            input_folder=input_folder,
            input_file_prefix=args.input_file_prefix,
            output_folder=output_folder
        )

    if args.compile:
        compile_into_csv(
            topk=args.topk,
            maxk=args.maxk,
            beam_size=args.beam_size,
            temperature=args.temperature,
            phases=phases,
            input_folder=input_folder,
            input_file_prefix=args.input_file_prefix,
            output_folder=output_folder
        )
 