try:
    from crem_updated import crem
except ImportError:
    from crem import crem

import os
import pickle
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesWriter
from tqdm import tqdm


def mol_smi_from_mol_pickle(input_pickle_filename: str = '50k_mol_smis',
                        output_smi_filename: str = '50k_mol_smis',
                        root: Optional[Union[str, bytes, os.PathLike]] = None):
    if root is None:
        root = Path(__file__).parents[2] / 'data' / 'cleaned_data' 

    with open(root / f'{input_filename}.pickle', 'rb') as handle:
        mol_smis = pickle.load(handle)
    
    with SmilesWriter(root / f'{output_filename}.smi') as writer:
        for mol_smi in tqdm(mol_smis):
            mol = Chem.MolFromSmiles(mol_smi)
            writer.write(mol)
 
def gen_crem_negs(num_neg: int,
                max_size: int,
                radius: int,
                frag_db: Union[str, bytes, os.PathLike,
                                sqlite3.Cursor, sqlite3.Connection],
                rxn_smi_file_prefix: Union[str, bytes, os.PathLike] = '50k_clean_rxnsmi_noreagent',
                dataset_name: Optional[str] = '50k', 
                root: Optional[Union[str, bytes, os.PathLike]] = None,
                splits: Optional[Union[str, 
                                        List[str]]] = None,
                ncores: Optional[int] = 1):
    '''     
    Assumes that there is only 1 product in every reaction!
    '''
    if splits is None:
        splits = ['train', 'valid', 'test']
    if root is None:
        root = Path(__file__).parents[1] / 'data' / 'cleaned_data'

    for split in splits:
        with open(root / f'{rxn_smi_file_prefix}_{split}.pickle', 'rb') as handle:
            rxn_smi_split = pickle.load(handle)
            
        all_mut_prod_smi = {} 
        insufficient = {} 
        for i, rxn_smi in enumerate(tqdm(rxn_smi_split[:20])):
            prod_smi = rxn_smi.split('>>')[-1]
            prod_mol = Chem.MolFromSmiles(prod_smi)
            this_rxn_mut = []
            j = 0 
            for j, mut_prod_smi in enumerate(crem.mutate_mol(prod_mol, db_name=frag_db, radius=radius, 
                                                max_size=max_size, return_mol=False, ncores=ncores)):
                this_rxn_mut.append(mut_prod_smi)
                j += 1
                if j > num_neg - 1:
                    break
            all_mut_prod_smi[prod_smi] = this_rxn_mut
            if j < num_neg - 1:
                # print(f'At index {i}, {j}<{num_neg}')
                insufficient[rxn_smi] = j 
        
        with open(root / f'{dataset_name}_neg{num_neg}_rad{radius}_maxsize{max_size}_{split}_mutprodsmis.pickle', 'wb') as handle:
            pickle.dump(all_mut_prod_smi, handle, pickle.HIGHEST_PROTOCOL)
        with open(root / f'{dataset_name}_neg{num_neg}_rad{radius}_maxsize{max_size}_{split}_insufficient.pickle', 'wb') as handle:
            pickle.dump(insufficient, handle, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # mol_smi_from_pickle('50k_mol_smis', '50k_mol_smis') # then run crem_create_frag_db.sh from command prompt
    gen_crem_negs(num_neg=50, max_size=3, radius=2, 
                frag_db='data/cleaned_data/replacements02_sa2.db')