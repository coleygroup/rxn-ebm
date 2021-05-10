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

sys.path.append('.') # to allow importing from rxnebm
try: #placeholder to just process proposals locally on windows w/o installing gln
    from rxnebm.proposer.gln_config import gln_config
    from rxnebm.proposer.gln_proposer import GLNProposer
except Exception as e:
    print(e)

def merge_chunks(
            topk: int = 50,
            maxk: int = 100,
            beam_size: int = 50,
            phase: str = 'train',
            start_idxs : List[int] = [0, 13000, 26000],
            end_idxs : List[int] = [13000, 26000, None], 
            output_folder: Optional[Union[str, bytes, os.PathLike]] = None
            ):
    """ Helper func to combine separatedly computed chunks into a single chunk, for the specified phase
    """
    merged = {} 
    logging.info(f'Merging start_idxs {start_idxs} and end_idxs {end_idxs}')
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        with open(output_folder / f'GLN_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}_start{start_idx}_end{end_idx}.pickle', 
        'rb') as handle:
            chunk = pickle.load(handle)
        for key, value in chunk.items(): 
            merged[key] = value
    
    with open(output_folder / f'GLN_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}.pickle', 'wb') as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f'Merged all chunks of {phase}!')
    return 

def gen_proposals(
            topk: int = 200,
            maxk: int = 200, 
            beam_size: Optional[int] = 200,
            phases: Optional[List[str]] = ['train', 'valid', 'test'],
            start_idx : Optional[int] = 0,
            end_idx : Optional[int] = None, 
            input_folder: Optional[Union[str, bytes, os.PathLike]] = None,
            input_file_prefix: Optional[str] = '50k_clean_rxnsmi_noreagent_allmapped_canon',
            output_folder: Optional[Union[str, bytes, os.PathLike]] = None,
            checkpoint_every: Optional[int] = 4000,
            model_path: Optional[Union[str, bytes, os.PathLike]] = "./rxnebm/proposer/GLN_original/dropbox/schneider50k.ckpt"
            ):
    '''
    Parameters
    ----------
    topk : int (Default = 50)
        for each product, how many proposals to put in train
    maxk : int (Default = 100)
        for each product, how many proposals to put in valid/test 
    beam_size : int (Default = 50)  
        beam size to use for ranking generated proposals
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
    gln_config["model_path"] = model_path
    arg_file = os.path.join(model_path, 'args.pkl')
    with open(arg_file, 'rb') as f:
        gln_config["args"] = pickle.load(f)
    gln_config["args"].model_path = model_path
    gln_config["args"].atom_file = gln_config["atom_file"]
    gln_config["args"].processed_data_path = gln_config["processed_data_path"]
    gln_config["args"].model_name = gln_config["model_name"]
    logging.info(f"gln_config: \n{gln_config}\n")

    proposer = GLNProposer(gln_config)

    clean_rxnsmis = {} 
    for phase in phases:
        with open(input_folder / f'{input_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmis[phase] = pickle.load(handle)

        phase_proposals = {} # key = prod_smi, value = Dict[template, reactants, scores]
        logging.info(f'Calculting for start_idx: {start_idx}, end_idx: {end_idx}')
        phase_topk = topk if phase == 'train' else maxk
        for i, rxn_smi in enumerate(
                                tqdm(
                                    clean_rxnsmis[phase][start_idx:end_idx], 
                                    desc=f'Generating GLN proposals for {phase}'
                                )
                            ):
            prod_smi_mapped = rxn_smi.split('>>')[-1]
            rxn_type = ["UNK"]

            curr_proposals = proposer.propose([prod_smi_mapped], rxn_type, topk=phase_topk, beam_size=beam_size)
            phase_proposals[prod_smi_mapped] = curr_proposals[0] # curr_proposals is a list w/ 1 element (which is a dict)

            if i > 0 and i % checkpoint_every == 0: # checkpoint
                logging.info(f'Checkpointing {i} for {phase}')
                with open(output_folder / f'GLN_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}_start{start_idx}_end{i + start_idx}.pickle', 'wb') as handle:
                    pickle.dump(phase_proposals, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if start_idx == 0 and end_idx is None:
            with open(output_folder / f'GLN_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}.pickle', 'wb') as handle:
                pickle.dump(phase_proposals, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_folder / f'GLN_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}_start{start_idx}_end{end_idx}.pickle', 'wb') as handle:
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
                beam_size: Optional[int] = 50,
                phases: Optional[List[str]] = ['train', 'valid', 'test'],
                input_folder: Optional[Union[str, bytes, os.PathLike]] = None,
                input_file_prefix: Optional[str] = '50k_clean_rxnsmi_noreagent_allmapped_canon',
                output_folder: Optional[Union[str, bytes, os.PathLike]] = None
                ):
    for phase in phases:
        logging.info(f'Processing {phase} of {phases}')

        if (output_folder / 
            f'GLN_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}.pickle'
        ).exists(): # file already exists
            with open(output_folder / 
                f'GLN_proposed_smiles_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}.pickle', 'rb'
            ) as handle:
                proposals_phase = pickle.load(handle) 
        else:
            raise RuntimeError('Error! Could not locate and load GLN proposed smiles file') 
        
        with open(input_folder / f'{input_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmi_phase = pickle.load(handle)
 
        proposed_precs_phase, prod_smiles_phase, rcts_smiles_phase = [], [], []
        proposed_precs_phase_withdups = [] # true representation of model predictions, for calc_accs() 
        prod_smiles_mapped_phase = [] # helper for analyse_proposed() 
        phase_topk = topk if phase == 'train' else maxk
        dup_count = 0
        for rxn_smi in tqdm(clean_rxnsmi_phase, desc='Processing rxn_smi'):     
            prod_smi = rxn_smi.split('>>')[-1]
            prod_mol = Chem.MolFromSmiles(prod_smi)
            [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
            prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
            # Sometimes stereochem takes another canonicalization...(more for reactants, but just in case)
            prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)
            prod_smiles_phase.append(prod_smi_nomap)
            prod_smiles_mapped_phase.append(prod_smi)
            
            # value for each prod_smi is a dict, where key = 'reactants' retrieves the predicted precursors
            precursors = proposals_phase[prod_smi]['reactants']
            
            # remove duplicate predictions
            seen = []
            for prec in precursors: # canonicalize all predictions
                prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec), True)
                if prec not in seen:
                    seen.append(prec)
                else:
                    dup_count += 1

            if len(seen) < phase_topk:
                seen.extend(['9999'] * (phase_topk - len(seen)))
            else:
                seen = seen[:phase_topk]
            proposed_precs_phase.append(seen)
            proposed_precs_phase_withdups.append(precursors)

            rcts_smi = rxn_smi.split('>>')[0]
            rcts_mol = Chem.MolFromSmiles(rcts_smi)
            [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
            rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
            # Sometimes stereochem takes another canonicalization...
            rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)
            rcts_smiles_phase.append(rcts_smi_nomap)
        dup_count /= len(clean_rxnsmi_phase)
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
        # if training data: remove ground truth prediction from proposals
        if phase == 'train':
            logging.info('\n(For training only) Double checking accuracy after removing ground truth predictions')
            _ = calc_accs(
                    [phase],
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    proposed_precs_phase
                )

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
        
        phase_dataframe = pd.DataFrame(
            temp_dataframe['zipped'].to_list(),
            index=temp_dataframe.index
        )
        
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
            f'GLN_{topk}topk_{maxk}maxk_{beam_size}beam_{phase}.csv',
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
        for n in [1, 3, 5, 10, 20, 50, 100, 200]:
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
        precursors = proposals_phase[mapped_key]['reactants']
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
    parser = argparse.ArgumentParser("gen_gln.py")
    parser.add_argument('-f') # filler for COLAB
    
    parser.add_argument("--log_file", help="log_file", type=str, default="gen_gln")

    parser.add_argument("--model_path", help="model checkpoint folder", type=str)
    parser.add_argument("--input_folder", help="input folder", type=str)
    parser.add_argument("--input_file_prefix", help="input file prefix of atom-mapped rxn smiles", type=str,
                        default="50k_clean_rxnsmi_noreagent_allmapped_canon")
    parser.add_argument("--output_folder", help="output folder", type=str)
    parser.add_argument("--location", help="location of script ['COLAB', 'LOCAL']", type=str, default="LOCAL")

    parser.add_argument("--propose", help='Whether to generate proposals (or just compile)', action="store_true")
    parser.add_argument("--train", help="Whether to generate and/or compile train preds", action="store_true")
    parser.add_argument("--valid", help="Whether to generate and/or compile valid preds", action="store_true")
    parser.add_argument("--test", help="Whether to generate and/or compile test preds", action="store_true")
    parser.add_argument("--start_idx", help="Start idx (train)", type=int, default=0)
    parser.add_argument("--end_idx", help="End idx (train)", type=int)
    parser.add_argument("--checkpoint_every", help="Save checkpoint of proposed smiles every N product smiles",
                        type=int, default=4000)

    parser.add_argument("--merge_chunks", help="Whether to merge already computed chunks", action="store_true")
    parser.add_argument("--phase_to_merge", help="Phase to merge chunks of (only supports phase at a time)", 
                        type=str, default='train')
    parser.add_argument("--chunk_start_idxs", help="Start idxs of computed chunks, separate by commas e.g. 0,10000,20000", 
                        type=str, default='0,15000,30000')
    parser.add_argument("--chunk_end_idxs", help="End idxs of computed chunks, separate by commas, for 'None'\
                        just type a comma not followed by any number e.g. 10000,20000,", type=str, default='15000,30000,')
    
    parser.add_argument("--compile", help="Whether to compile proposed precursor SMILES (& rxn_smiles data) into CSV file", 
                        action='store_true')

    parser.add_argument("--beam_size", help="Beam size", type=int, default=200)
    parser.add_argument("--topk", help="How many top-k proposals to put in train (not guaranteed)", type=int, default=200)
    parser.add_argument("--maxk", help="How many top-k proposals to generate and put in valid/test (not guaranteed)", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args() 

    RDLogger.DisableLog("rdApp.warning")

    os.makedirs(Path(__file__).resolve().parents[1] / "logs/gen_gln/", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(Path(__file__).resolve().parents[1] / f"logs/gen_gln/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.input_folder is None:
        input_folder = Path(__file__).resolve().parents[1] / 'rxnebm/data/cleaned_data/'
        print(input_folder)
    else:
        input_folder = Path(args.input_folder)
    if args.output_folder is None:
        if args.location == 'COLAB':
            output_folder = Path('/content/gdrive/MyDrive/rxn_ebm/datasets/Retro_Reproduction/GLN_proposals/')
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_folder = input_folder
    else:
        output_folder = Path(args.output_folder)
        os.makedirs(output_folder, exist_ok=True)

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

    if args.propose:
        gen_proposals(
            maxk=args.maxk,
            topk=args.topk,
            beam_size=args.beam_size,
            phases=phases,
            start_idx=args.start_idx,
            end_idx=args.end_idx, 
            input_folder=input_folder,
            input_file_prefix=args.input_file_prefix,
            output_folder=output_folder,
            checkpoint_every=args.checkpoint_every,
            model_path=args.model_path
        ) 

    if args.merge_chunks:
        merge_chunks(
            topk=args.topk,
            maxk=args.maxk,
            beam_size=args.beam_size,
            phase=args.phase_to_merge, 
            start_idxs=[int(num) if num != '' else None for num in args.chunk_start_idxs.split(',')], # [0, 12000, 15000, 30000] 
            end_idxs=[int(num) if num != '' else None for num in args.chunk_start_idxs.split(',')], # [12000, 15000, 30000, None], 
            output_folder=output_folder
        )

    if args.compile:
        compile_into_csv(
            topk=args.topk,
            maxk=args.maxk,
            beam_size=args.beam_size,
            phases=phases,
            input_folder=input_folder,
            input_file_prefix=args.input_file_prefix,
            output_folder=output_folder
        )
    