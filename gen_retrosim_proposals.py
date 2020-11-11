import pandas as pd
import numpy as np
from tqdm import tqdm  
import os
import time 
from pathlib import Path 
import gc
gc.enable()
from scipy import sparse

from rxnebm.data import augmentors
from rxnebm.data.preprocess import smi_to_fp

def main(): #TODO: add and parse args 
    ''' Takes a while! about 40 min for train, 5 min each for valid & test. 
    ''' 

    proposals_file_prefix = "retrosim_200maxtest_200maxprec"
    cleaned_data_root = None
    radius = 3
    fp_size = 4096
    dtype = 'int32'
    max_valid_neg = 200

    if cleaned_data_root is None:
        cleaned_data_root = Path.cwd().resolve() / 'rxnebm/data/cleaned_data'

    phases_to_compute = ['train', 'valid', 'test']
    for phase in phases_to_compute:
        print(f'Processing {phase}')
        
        proposals = pd.read_csv(cleaned_data_root / f'{proposals_file_prefix}_{phase}.csv', index_col=None)
        proposals = proposals.fillna('9999') 
        proposals_trimmed = proposals.drop(['Unnamed: 0', 'orig_rxn_smi', 'rank_of_true_precursor'], axis=1)
        proposals_numpy = proposals_trimmed.values
        
        max_col = float('-inf') # marker to keep track of longest # of cols needed in sparse matrix
        for row_idx, row in enumerate(proposals_numpy):
            for i, precursor in enumerate(row[1:]):
                if str(precursor) == '9999':
                    this_max_col = i
                    break 
    #         if max_col < this_max_col:
    #             print(f'At row {row_idx}, max_col = {this_max_col}')
            max_col = max(this_max_col, max_col)
        print(f'Checked max non N/A columns! Found max non N/A col: {max_col}')

        processed_rxn_smis = []
        for row in proposals_numpy:
            neg_rxn_smis = []
            pos_rxn_smi = row[1] + '>>' + row[0] # true_precursors>>prod_smi 
            for neg_precursor in row[2:]:
                # this neg_proposal and later ones in the same row are all N/A, therefore break the loop 
                if str(neg_precursor) == '9999': 
                    break
                elif neg_precursor == row[1]: # this shouldn't happen, since we already filtered out all true_precursors
                    raise RuntimeError()
                else:
                    neg_rxn_smi = neg_precursor + '>>' + row[0] #neg_precursors>>prod_smi 
                    neg_rxn_smis.append(neg_rxn_smi)

            processed_rxn_smis.append([pos_rxn_smi, *neg_rxn_smis])
        print('Processed into pos and neg rxn_smis!')
            
        uniq_mol_smis = set() 
        for rxn in processed_rxn_smis:
            for rxn_smi in rxn:
                rcts = rxn_smi.split(">")[0]
                prod = rxn_smi.split(">")[-1] 
                rcts_prod_smis = rcts + "." + prod
                for mol_smi in rcts_prod_smis.split("."):
                    if mol_smi == '':
                        # print(f'At index {i} of {phase}, found mol_smis == ""')
                        continue 
                    uniq_mol_smis.add(mol_smi)

        uniq_mol_smis = list(uniq_mol_smis)
        print('Generated unique mol_smis!')
        time.sleep(1)

        count_mol_fps = [] # bad_idx = []
        for i, mol_smi in enumerate(tqdm(uniq_mol_smis, total=len(uniq_mol_smis), desc='Generating count fps...')):
            cand_mol_fp = smi_to_fp.mol_smi_to_count_fp(mol_smi, radius, fp_size, dtype)
    #         if cand_mol_fp.nnz == 0:
    #             raise ValueError('Error, one of the mol_smi gave all-zero count fingerprint. Debug!')
    #             # bad_idx.append(i) 
    #             # continue
    #         else:  # sparsify then append to list
            count_mol_fps.append(cand_mol_fp)
                
        mol_smi_to_fp = {} 
        for i, mol_smi in enumerate(uniq_mol_smis):
            mol_smi_to_fp[mol_smi] = i

        phase_rxn_fps = []
        for rxn in tqdm(processed_rxn_smis, desc='Generating rxn_fps...'):
            pos_rxn_smi = rxn[0]
            neg_rxn_smis = rxn[1:]

            pos_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(pos_rxn_smi, 'count', mol_smi_to_fp, count_mol_fps)
            pos_rxn_fp = augmentors.make_rxn_fp(pos_rcts_fp, prod_fp, 'diff')

            neg_rxn_fps = []
            for neg_rxn_smi in neg_rxn_smis:
                neg_rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(neg_rxn_smi, 'count', mol_smi_to_fp, count_mol_fps)
                neg_rxn_fp = augmentors.make_rxn_fp(neg_rcts_fp, prod_fp, 'diff')
                neg_rxn_fps.append(neg_rxn_fp)

            if len(neg_rxn_fps) < max_col:
                dummy_fp = np.zeros((1, fp_size))
                neg_rxn_fps.extend([dummy_fp] * (max_col - len(neg_rxn_fps))) 

            this_rxn_fps = sparse.hstack([pos_rxn_fp, *neg_rxn_fps])
            phase_rxn_fps.append(this_rxn_fps)

        # TODO: don't use sparse.vstack, it's memory intensive and will OOM for larger files
        phase_rxn_fps = sparse.vstack(phase_rxn_fps)
        phase_rxn_fps = phase_rxn_fps.tocsr() 

        sparse.save_npz(cleaned_data_root / f'retrosim_rxn_fps_{phase}.npz', phase_rxn_fps)

if __name__ == '__main__':
    main() 