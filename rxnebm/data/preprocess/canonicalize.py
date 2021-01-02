import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import rdkit
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm

# TODO: merge this w/ script to generate clean_csv for training GLN from scratch
# i.e. save output as both .pickle & .csv files
def canonicalize_rxn_smi(
    rxn_smi: str, 
    remove_mapping: bool = False
    ):
    prod_smi = rxn_smi.split('>>')[-1]
    prod_mol = Chem.MolFromSmiles(prod_smi)

    rcts_smi = rxn_smi.split('>>')[0]
    rcts_mol = Chem.MolFromSmiles(rcts_smi)
    
    if remove_mapping:
        [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
        prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
        # double canonicalize
        prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)

        [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
        rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
        # double canonicalize - sometimes stereochem takes another canonicalization...
        rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)

        rct_changed = 1 if rcts_smi_nomap != rcts_smi else 0        
        prod_changed = 1 if prod_smi_nomap != prod_smi else 0

        return rcts_smi_nomap + '>>' + prod_smi_nomap, rct_changed, prod_changed
    else:
        prod_smi_map = Chem.MolToSmiles(prod_mol, True)
        # double canonicalize
        prod_smi_map = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_map), True)
        
        rcts_smi_map = Chem.MolToSmiles(rcts_mol, True)
        # double canonicalize
        rcts_smi_map = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_map), True)

        rct_changed = 1 if rcts_smi_map != rcts_smi else 0        
        prod_changed = 1 if prod_smi_map != prod_smi else 0

        return rcts_smi_map + '>>' + prod_smi_map, rct_changed, prod_changed

def canonicalize_phases(
    phases: List[str] = ['train', 'valid', 'test'],
    remove_mapping: bool = False,
    input_data_file_prefix: str = '50k_clean_rxnsmi_noreagent_allmapped',
    input_data_folder: Optional[Union[str, bytes, os.PathLike]] = None,
    ):
    ''' 
    input_data_file should have atom-mapped reaction SMILES 
    '''
    clean = {}
    rxn_smis_canon = {}
    for phase in phases:
        phase_rxn_smis_canon = []
        rct_changed, prod_changed = 0, 0
        
        if input_data_folder is None:
            input_data_folder = Path(__file__).resolve().parents[2] / 'data/cleaned_data'
        else:
            input_data_folder = Path(input_data_folder)
        with open(input_data_folder / f'{input_data_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean[phase] = pickle.load(handle) 

        for rxn_smi in tqdm(clean[phase], desc=f'Processing {phase}'):
            rxn_smi_canon, r_change, p_change = \
                canonicalize_rxn_smi(rxn_smi, remove_mapping=remove_mapping)
            rct_changed += r_change
            prod_changed += p_change
 
            phase_rxn_smis_canon.append(rxn_smi_canon)

        rxn_smis_canon[phase] = phase_rxn_smis_canon

        # remove duplicates
        dups = 0
        rxn_smis_seen, rxn_smis_nodup = set(), [] # use list to preserve order
        for rxn_smi_canon in phase_rxn_smis_canon:
            if rxn_smi_canon not in rxn_smis_seen:
                rxn_smis_seen.add(rxn_smi_canon)
                rxn_smis_nodup.append(rxn_smi_canon)
            else:
                dups += 1

        print('\n', '#'*10, f'Stats for {phase}', '#'*10)
        print(f'rxn_smi_canon total: {len(phase_rxn_smis_canon)}')
        print(f'rxn_smi_canon duplicates: {dups}')
        print(f'prod_smi mismatch: {prod_changed}')
        print(f'rcts_smi mismatch: {rct_changed}')

        with open(input_data_folder / f'{input_data_file_prefix}_canon_{phase}.pickle', 'wb') as handle:
            pickle.dump(rxn_smis_nodup, handle, protocol=4) # 4 compatible w/ py36

if __name__ == '__main__':
    from rxnebm.data.preprocess import clean_smiles

    input_prefixes = [
        '50k_clean_rxnsmi_noreagent_allmapped',
        '50k_clean_rxnsmi_noreagent'
    ]
    remove_mapping = [False, True]
    for prefix, remove in zip(input_prefixes, remove_mapping):
        canonicalize_phases(input_data_file_prefix=prefix, remove_mapping=remove)
        clean_smiles.remove_overlapping_rxn_smis(rxn_smi_file_prefix=f'{prefix}_canon')