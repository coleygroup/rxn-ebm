import sys
import os

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdChemReactions
from rdkit.Chem import rdmolops
from rdkit import DataStructs
import numpy as np

from itertools import chain, compress

from tqdm import tqdm
import csv
import re 
import pickle

from typing import List

def get_rxn_smi(mol_prod: rdkit.Chem.rdchem.Mol, reactants: List[rdkit.Chem.rdchem.Mol], 
                  original_reagents: List[str], keep_reagents: bool = True, remove_mapping: bool = True):
    '''
    Adapted func from Hanjun Dai's GLN to additionally keep track of reagents (reactants that don't appear in products)
    Gets rid of reactants when they don't contribute to the product

    Parameters
    ----------
    mol_prod : rdkit.Chem.rdchem.Mol
        product molecule     
    reactants : List[rdkit.Chem.rdchem.Mol]
        list of reactant molecules 
    original_reagents: List[str]
        list of reagents in OriginalReaction, each element of list = 1 reagent
    keep_reagents: bool (Default = True)
        whether to keep reagents in the output SMILES string
    remove_mapping: bool (Default = True)
        whether to remove atom mapping if atom in rct is not in prod

    Returns
    -------
    rxn_smi : str
        reaction SMILES string with only reactants that contribute to the product, 
        the other molecules being part of reagents (if keep_reagent)
    '''
    prod_smi = Chem.MolToSmiles(mol_prod, True)
    prod_maps = set(re.findall('\:([[0-9]+)\]', prod_smi))
    reactants_smi_list = []
    
    reagent_smi_list = []
    if original_reagents:
        reagent_smi_list.append(original_reagents)
    
    for mol in reactants:
        if mol is None:
            continue
            
        used = False
        for a in mol.GetAtoms():
            if a.HasProp('molAtomMapNumber'):
                if a.GetProp('molAtomMapNumber') in prod_maps:
                    used = True 
                    
                elif remove_mapping: # removes atom mapping if atom in reactant is not in product
                    a.ClearProp('molAtomMapNumber')
                    
        if used:
            reactants_smi_list.append(Chem.MolToSmiles(mol, True))
        else:
            reagent_smi_list.append(Chem.MolToSmiles(mol, True))
            
    reactants_smi = '.'.join(reactants_smi_list)
    
    if not keep_reagent:
        return '{}>>{}'.format(reactants_smi, prod_smi)

    if reagent_smi_list:
        reagents_smi = '.'.join(reagent_smi_list)
    else: # reagent_smi_list is empty
        reagents_smi = ''
    return '{}>{}>{}'.format(reactants_smi, reagents_smi, prod_smi)


def clean_data(base_path_rxnsmi: str, dataset: str, keep_reagent: bool = True, remove_mapping: bool = True):
    '''
    Cleans reaction SMILES strings by removing those with: 
        bad product (SMILES not parsable by rdkit)
    
    It also checks these, but does not remove them (since atom mapping is not important for us):
        missing atom mapping (not all atoms in the product molecule have atom mapping), 
        bad atom mapping (not 1:1 between reactants and products)

    Parameters
    ----------
    base_path_rxnsmi : str 
        base path to the CSV file containing the reaction SMILES strings 
        e.g. 'original_data/schneider50k/raw' 
        internally, '_{dataset}.csv' will be appended 
    dataset : str 
        to be looped between 'train', 'valid', 'test'
    keep_reagent : bool (Default = True)
        whether to keep reagents in the output SMILES
    remove_mapping : bool (Default = True)
        whether to remove atom mapping if atom in rct is not in prod

    Returns
    -------
    clean_list : List[str]
        list of cleaned reaction SMILES strings

    Also see: get_rxn_smi 
    TODO: autodetect which row to correctly start parsing from (it's 1 for USPTO_50k, but 3 for USPTO_STEREO)
    '''
    filename = str(base_path_rxnsmi) + f'_{dataset}.csv'
    print('Processing {}'.format(dataset))
    
    pt = re.compile(r':(\d+)]')
    clean_list = [] # to store cleaned rxn smiles
    # bad_mapping = 0
    bad_prod = 0
    # missing_map = 0
    raw_num = 0
    
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader) # pass first row of csv file (for USPTO_STEREO, 3 lines to be passed)
        
        for row in reader:
            rxn_smiles = row[0].split(',')[2] 
            all_reactants, reagents, prod = rxn_smiles.split('>')

            rids = ','.join(sorted(re.findall(pt, all_reactants)))
            pids = ','.join(sorted(re.findall(pt, prod)))
            if rids != pids:  # atom mapping is not 1:1
                bad_mapping += 1
                # continue

            reactants = [Chem.MolFromSmiles(smi) for smi in all_reactants.split('.')]

            mol_prod = Chem.MolFromSmiles(prod)
            if mol_prod is None:  # rdkit is not able to parse the product
                bad_prod += 1
                continue
                
            # Make sure all have atom mapping
            if not all([a.HasProp('molAtomMapNumber') for a in mol_prod.GetAtoms()]):
                missing_map += 1
                # continue

            raw_num += 1
            rxn_smiles = get_rxn_smi(mol_prod, reactants, reagents, keep_reagent, remove_mapping)

            clean_list.append(rxn_smiles)  

            if raw_num%10000 == 0:
                print('Extracted: {} raw rxn'.format(raw_num))
        
        print('# clean rxn:', len(clean_list))
        # print('# unique rxn:', len(set(clean_list))) 
        print('bad mapping:', bad_mapping)
        print('bad prod:', bad_prod)
        print('missing map:', missing_map)
        print('raw rxn extracted:', raw_num, '\n')
        return clean_list

def main():
    from pathlib import Path   

    root = Path(__file__).parents[1]
    path_prefix = 'original_data/schneider50k/raw'
    clean_rxn_smi_output_prefix = 'clean_rxn_50k_nomap_noreagent'

    clean_rxn = {'train': None, 'valid': None, 'test': None}
    for dataset in clean_rxn.keys():
        clean_rxn[dataset] = clean_data(root / path_prefix , dataset, keep_reagents=False, remove_mapping=True, 
                                )       
        
        output_filepath = str(root / 'cleaned_data/') + clean_rxn_smi_output_prefix + f'_{dataset}.pickle'
        with open(output_filepath, 'wb') as handle:
            pickle.dump(clean_rxn[dataset], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    mol_smis = set()
    for key in clean_rxn.keys(): 
        [mol_smis.update(mol_smi) for rxn_smi in tqdm(clean_rxn[key]) 
            for half_rxn_smi in rxn_smi.split('>>') 
            for mol_smi in [half_rxn_smi.split('.')]]
    
    uniq_mol_smi_output_prefix = '50k_unique_rcts_prod_smis.pickle'
    with open(root / 'cleaned_data' / uniq_mol_smi_output_prefix, 'wb') as handle:
        pickle.dump(list(mol_smis), handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()