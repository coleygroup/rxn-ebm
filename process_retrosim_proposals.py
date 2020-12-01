import argparse
import logging 
import os
import sys
import time 
import gc
gc.enable()

from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm  
from pathlib import Path 
from scipy import sparse
from rdkit import RDLogger

from rxnebm.data import augmentors
from rxnebm.data.preprocess import smi_to_fp
from rxnebm.proposer import retrosim_model

''' TODO: add support for representation: ['Graphs'] for transformer/GNNs
'''

def parse_args():
    parser = argparse.ArgumentParser("process_retrosim_proposals.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="")
    parser.add_argument("--cleaned_data_root", help="Path to cleaned_data folder (do not change)", type=str,
                        default=None) 
    parser.add_argument("--rxn_smi_file_prefix", help="Prefix of the 3 pickle files containing the train/valid/test reaction SMILES strings (do not change)", type=str,
                        default='50k_clean_rxnsmi_noreagent_allmapped') 
    parser.add_argument("--output_file_prefix", help="Prefix of 3 output files containing Retrosim's proposals in chosen representation", type=str,
                        default='retrosim_rxn_fps') # add graphs to prefix when necessary

    # Retrosim arguments (rxnebm/proposer/retrosim_model.py)
    parser.add_argument("--topk", help="No. of proposals Retrosim should try to generate per product SMILES (not guaranteed)", type=int, default=200)
    parser.add_argument("--max_precedents", help="No. of similar products Retrosim should compare against", type=int, default=200)
    parser.add_argument("--similarity_type", 
        help="Metric to use for similarity search of product fingerprints ['Tanimoto', 'Dice', 'TverskyA', 'TverskyB']", 
        type=str, default="Tanimoto")
    parser.add_argument("--retrosim_fp_type", 
        help="Fingerprint type to use ['Morgan2noFeat', 'Morgan3noFeat', 'Morgan2Feat', 'Morgan3Feat']", 
        type=str, default="Morgan2Feat")
    parser.add_argument("--parallelize", help="Whether to parallelize the proposal generation step", action="store_true")

    # TODO: add graphs 
    parser.add_argument("--representation", help='the representation to compute ["fingerprints", "rxn_smiles"]', default='fingerprints')
    # fingerprint arguments, only apply if --representation=='fingerprints'
    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=3)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=4096)
    parser.add_argument("--rxn_type", help='Fingerprint reaction type ["diff", "sep"]', type=str, default="diff")
    parser.add_argument("--fp_type", help='Fingerprint type ["count", "bit"]', type=str, default="count")

    return parser.parse_args()
 
def main(args):
    '''
    1. Generate retrosim proposals for all product SMILES in USPTO_50k, takes a while: about ~20-25 hours on an 8-core machine
    2. Process all proposed SMILES into fingerprints, also takes some time: about 40 min for train, 5 min each for valid & test, for 4096-dim count, diff fingerprints
    '''
    # retrieve args 
    max_prec = args.max_precedents
    topk = args.topk
    similarity_type = args.similarity_type
    retrosim_fp_type = args.retrosim_fp_type
    parallelize = args.parallelize
    rxn_smi_file_prefix = args.rxn_smi_file_prefix
    proposals_file_prefix = f"retrosim_{topk}maxtest_{max_prec}maxprec" # do not change

    cleaned_data_root = args.cleaned_data_root
    output_file_prefix = args.output_file_prefix
    representation = args.representation
    radius = args.radius
    fp_size = args.fp_size
    rxn_type = args.rxn_type
    fp_type = args.fp_type
    dtype = 'int32' # do not change

    if cleaned_data_root is None:
        cleaned_data_root = Path(__file__).parents[0].resolve() / 'rxnebm/data/cleaned_data'

    for phase in ['train', 'valid', 'test']:
        if not (cleaned_data_root / f'{proposals_file_prefix}_{phase}.csv').exists():
            logging.info('Could not find proposals CSV file by retrosim; generating them now! (Can take up to 25 hours!!!)')
            retrosim_model = retrosim_model.Retrosim(topk=topk, max_prec=max_prec, similarity_type=similarity_type,
                                    input_data_file_prefix=rxn_smi_file_prefix, fp_type=retrosim_fp_type, parallelize=parallelize)
            retrosim_model.prep_valid_and_test_data()
            retrosim_model.propose_all()
            retrosim_model.analyse_proposed() 
            break # propose_all() settles all 3 of train, valid & test

    # check if ALL 3 files to be computed already exist in cleaned_data_root, if so, then just exit the function
    phases_to_compute = ['train', 'valid', 'test'] 
    for phase in ["train", "valid", "test"]:
        if (cleaned_data_root / f"{output_file_prefix}_{phase}.npz").exists() or\
            (cleaned_data_root / f"{output_file_prefix}_{phase}.pickle").exists():
            print(f"At: {cleaned_data_root / output_file_prefix}_{phase}")
            print("The processed retrosim_rxn_fps file already exists!")
            phases_to_compute.remove(phase)        
    if len(phases_to_compute) == 0: # all pre-computed, OK 
        return
    elif len(phases_to_compute) < 3: # corrupted 
        raise RuntimeError('The retrosim proposal processing is most likely corrupted. Please delete existing files and restart!')
    for phase in phases_to_compute:
        logging.info(f'Processing {phase}')
        
        proposals = pd.read_csv(cleaned_data_root / f'{proposals_file_prefix}_{phase}.csv', index_col=None, dtype='str')
        proposals = proposals.fillna('9999') 
        proposals_trimmed = proposals.drop(['orig_rxn_smi', 'rank_of_true_precursor'], axis=1)
        proposals_numpy = proposals_trimmed.values

        processed_rxn_smis = []
        for row in proposals_numpy:
            neg_rxn_smis = []
            pos_rxn_smi = row[1] + '>>' + row[0] # true_precursors>>prod_smi 
            for neg_precursor in row[2:]:
                # this neg_proposal and later ones in the same row are all N/A, therefore break the loop 
                if str(neg_precursor) == '9999': 
                    break
                else:
                    neg_rxn_smi = neg_precursor + '>>' + row[0] #neg_precursors>>prod_smi 
                    neg_rxn_smis.append(neg_rxn_smi)

            processed_rxn_smis.append([pos_rxn_smi, *neg_rxn_smis])
        logging.info('Processed into pos and neg rxn_smis!')

        if representation == 'rxn_smiles':
            with open(cleaned_data_root / f'retrosim_rxn_smis_{phase}.pickle', 'rb') as handle:
                pickle.dump(processed_rxn_smis, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'Saved {phase} rxn_smi!')

        elif representation == 'fingerprints':
            uniq_mol_smis = set() 
            for rxn in processed_rxn_smis:
                for rxn_smi in rxn:
                    rcts = rxn_smi.split(">")[0]
                    prod = rxn_smi.split(">")[-1] 
                    rcts_prod_smis = rcts + "." + prod
                    for mol_smi in rcts_prod_smis.split("."):
                        if mol_smi == '': 
                            continue 
                        uniq_mol_smis.add(mol_smi)

            uniq_mol_smis = list(uniq_mol_smis)
            logging.info('Generated unique mol_smis!')
            time.sleep(0.5)

            count_mol_fps = [] # bad_idx = []
            for i, mol_smi in enumerate(tqdm(uniq_mol_smis, total=len(uniq_mol_smis), desc='Generating mol fps...')):
                cand_mol_fp = smi_to_fp.mol_smi_to_count_fp(mol_smi, radius, fp_size, dtype)
                count_mol_fps.append(cand_mol_fp)
                    
            mol_smi_to_fp = {} 
            for i, mol_smi in enumerate(uniq_mol_smis):
                mol_smi_to_fp[mol_smi] = i
    
            phase_rxn_fps = [] # sparse.csr_matrix((len(proposals_file_prefix), fp_size)) # init big output sparse matrix
            if phase == 'train':
                for rxn in tqdm(processed_rxn_smis, desc='Generating rxn fps...'):
                    pos_rxn_smi = rxn[0]
                    neg_rxn_smis = rxn[1:]

                    pos_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(pos_rxn_smi, fp_type, mol_smi_to_fp, count_mol_fps)
                    pos_rxn_fp = augmentors.make_rxn_fp(pos_rcts_fp, prod_fp, rxn_type)

                    neg_rxn_fps = []
                    for neg_rxn_smi in neg_rxn_smis:
                        neg_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(neg_rxn_smi, fp_type, mol_smi_to_fp, count_mol_fps)
                        neg_rxn_fp = augmentors.make_rxn_fp(neg_rcts_fp, prod_fp, rxn_type)
                        neg_rxn_fps.append(neg_rxn_fp)

                    if len(neg_rxn_fps) < topk:
                        if rxn_type == 'sep' or rxn_type == 'hybrid':
                            dummy_fp = np.zeros((1, fp_size * 2))
                        elif rxn_type == 'hybrid_all':
                            dummy_fp = np.zeros((1, fp_size * 3))
                        else: # diff 
                            dummy_fp = np.zeros((1, fp_size))
                        neg_rxn_fps.extend([dummy_fp] * (topk - len(neg_rxn_fps))) 

                    this_rxn_fps = sparse.hstack([pos_rxn_fp, *neg_rxn_fps])
                    # phase_rxn_fps[i] = sparse.hstack([pos_rxn_fp, *neg_rxn_fps]) # significantly slower! 
                    phase_rxn_fps.append(this_rxn_fps)
            else: # do not assume pos_rxn_smi is inside 
                for rxn in tqdm(processed_rxn_smis, desc='Generating rxn_fps...'):
                    rxn_smis = rxn[1:] 

                    rxn_fps = []
                    for rxn_smi in rxn_smis:
                        rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(rxn_smi, fp_type, mol_smi_to_fp, count_mol_fps)
                        rxn_fp = augmentors.make_rxn_fp(rcts_fp, prod_fp, rxn_type)
                        rxn_fps.append(rxn_fp)

                    if len(rxn_fps) < topk:
                        if rxn_type == 'sep' or rxn_type == 'hybrid':
                            dummy_fp = np.zeros((1, fp_size * 2))
                        elif rxn_type == 'hybrid_all':
                            dummy_fp = np.zeros((1, fp_size * 3))
                        else: # diff 
                            dummy_fp = np.zeros((1, fp_size))
                        rxn_fps.extend([dummy_fp] * (topk - len(rxn_fps))) 

                    this_rxn_fps = sparse.hstack(rxn_fps)
                    phase_rxn_fps.append(this_rxn_fps)

            phase_rxn_fps = sparse.vstack(phase_rxn_fps)
            sparse.save_npz(cleaned_data_root / f"{output_file_prefix}_{phase}.npz", phase_rxn_fps)
        
        else:
            raise ValueError(f'{representation} not supported!')

if __name__ == '__main__':
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.propagate = False
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    for fp_size in [512, 1024, 2048, 4096*2, 4096*4]:
        args.fp_size = fp_size
        args.output_file_prefix = f"retrosim_rxn_fps_{fp_size}"
        logging.info(f'\nProcessing retrosim proposals into {fp_size}-dim diff fingerprints')
        main(args) 

    for fp_size in [1024, 4096, 4096*4]:
        args.fp_size = fp_size
        args.rxn_type = 'sep'
        args.output_file_prefix = f"retrosim_rxn_fps_{fp_size}_{args.rxn_type}"
        logging.info(f'Processing retrosim proposals into {fp_size}-dim sep fingerprints\n')
        main(args) 

    for fp_size in [1024, 4096, 4096*2, 4096*4]:
        args.fp_size = fp_size
        args.rxn_type = 'hybrid'
        args.output_file_prefix = f"retrosim_rxn_fps_{fp_size}_{args.rxn_type}"
        logging.info(f'Processing retrosim proposals into {fp_size}-dim hybrid fingerprints\n')
        main(args) 

    for fp_size in [4096*4, 4096*2, 4096]:
        args.fp_size = fp_size
        args.rxn_type = 'hybrid_all'
        args.output_file_prefix = f"retrosim_rxn_fps_{fp_size}_{args.rxn_type}"
        logging.info(f'Processing retrosim proposals into {fp_size}-dim hybrid_all fingerprints\n')
        main(args) 

    logging.info(f'Successfully prepared {args.representation} from retrosim proposals!')