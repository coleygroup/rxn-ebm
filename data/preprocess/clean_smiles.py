''' This module contains functions to clean a raw dataset of reaction SMILES strings, which includes
operations such as removing atom mapping and checking for invalid molecule SMILES. There is also 
a function to generate a list of SMILES strings of all unique molecules in the given dataset. Note that
in rxn-ebm, a dataset refers to the combination of 'train', 'valid', and 'test', e.g. USPTO_50k is a dataset.
'''
import sys
import os
from concurrent.futures import ProcessPoolExecutor as Pool

import rdkit
from rdkit import Chem 
from rdkit.Chem import rdChemReactions

import numpy as np
from tqdm import tqdm
import csv
import re 
import pickle
from pathlib import Path   
from typing import List, Optional

def remove_mapping(rxn_smi: str, keep_reagents: bool = False) -> str :
    ''' 
    NOTE: assumes only 1 product  
    Removes all atom mapping from the reaction SMILES string
    '''
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smi, useSmiles=True)
    if not keep_reagents:
        rxn.RemoveAgentTemplates()

    prod = [mol for mol in rxn.GetProducts()]
    for atom in prod[0].GetAtoms(): 
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber') 
    
    rcts = [mol for mol in rxn.GetReactants()]
    for rct in rcts:
        for atom in rct.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')

    return rdChemReactions.ReactionToSmiles(rxn)

def move_reagents(mol_prod: rdkit.Chem.rdchem.Mol, reactants: List[rdkit.Chem.rdchem.Mol], 
                            original_reagents: List[str], 
                            keep_reagents: bool = False, remove_rct_mapping: bool = True) -> str :
    '''
    NOTE: assumes only 1 product
    Adapted func from Hanjun Dai's GLN to additionally keep track of reagents (reactants that don't appear in products)
    Gets rid of reactants when they don't contribute to the product

    Parameters
    ----------
    mol_prod : rdkit.Chem.rdchem.Mol
        product molecule     
    reactants : List[rdkit.Chem.rdchem.Mol]
        list of reactant molecules 
    original_reagents : List[str]
        list of reagents in the original reaction, each element of list = 1 reagent
    keep_reagents : bool (Default = True)
        whether to keep reagents in the output SMILES string
    remove_rct_mapping : bool (Default = True)
        whether to remove atom mapping if atom in reactant is not in product (i.e. leaving groups)

    Returns
    -------
    str
        reaction SMILES string with only reactants that contribute to the product, 
        the other molecules being moved to reagents (if keep_reagents is True)
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
                    
                elif remove_rct_mapping: # removes atom mapping if atom in reactant is not in product
                    a.ClearProp('molAtomMapNumber')
                    
        if used:
            reactants_smi_list.append(Chem.MolToSmiles(mol, True))
        else:
            reagent_smi_list.append(Chem.MolToSmiles(mol, True))
            
    reactants_smi = '.'.join(reactants_smi_list)
    
    if not keep_reagents:
        return '{}>>{}'.format(reactants_smi, prod_smi)

    if reagent_smi_list:
        reagents_smi = '.'.join(reagent_smi_list)
    else: # reagent_smi_list is empty
        reagents_smi = ''
    return '{}>{}>{}'.format(reactants_smi, reagents_smi, prod_smi)

def clean_rxn_smis_from_csv(path_to_rxn_smis: str, lines_to_skip: int = 1, dataset: str='50k', 
                            keep_reagents: bool = False, remove_rct_mapping: bool = True,
                            remove_all_mapping: bool = True):
    '''
    NOTE: assumes only 1 product
    TODO: need to read csv file twice, first to get total line count, second to do the actual work
    This might not be practical with larger csv files 

    Cleans reaction SMILES strings by removing those with: 
        bad product (SMILES not parsable by rdkit)
        too small products (a large reactant fails to be recorded as a produc)
    
    It also checks these, but does not remove them, since atom mapping is not critical for us:
        missing atom mapping (not all atoms in the product molecule have atom mapping), 
        bad atom mapping (not 1:1 between reactants and products)

    Parameters
    ----------
    path_to_rxn_smis : str 
        full path to the CSV file containing the reaction SMILES strings
        there will be one CSV file each for train, valid and test, coordinated by clean_rxn_smis_all_datasets
    lines_to_skip : int (Default = 1)
        how many header lines to skip in the CSV file 
        This is 1 for USPTO_50k (schneider), but 3 for USPTO_STEREO 
        Unfortunately, this cannot be reliability automated, as every CSV file can be differently formatted
    keep_reagents : bool (Default = False)
        whether to keep reagents in the output SMILES
    remove_rct_mapping : bool (Default = True)
        whether to remove atom mapping if atom in reactant is not in product (i.e. leaving groups)
    remove_all_mapping : bool (Default = True)
        whether to remove all atom mapping from the reaction SMILES, 
        if True, remove_rct_mapping will be automatically set to True 

    Returns
    -------
    clean_list : List[str]
        list of cleaned reaction SMILES strings

    Also see: move_reagents, remove_mapping, clean_rxn_smis_from_file
    ''' 
    if remove_all_mapping:
        remove_rct_mapping = True 
    pt = re.compile(r':(\d+)]')
    clean_list = [] 
    bad_mapping = 0
    bad_prod = 0
    missing_map = 0
    raw_num = 0
    too_small = 0
    
    with open(path_to_rxn_smis, 'r') as csv_file:
        total_lines = len(csv_file.readlines()) - lines_to_skip
    
    with open(path_to_rxn_smis, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        for line in range(lines_to_skip): 
            next(reader) # pass first row of csv file (for USPTO_STEREO, 3 lines to be passed)
        
        for row in tqdm(reader, total=total_lines):
            if dataset == '50k':
                rxn_smi = row[0].split(',')[2] # second column of the csv file
                all_rcts_smi, all_reag_smi, prod_smi = rxn_smi.split('>')
            elif dataset == 'STEREO':
                rxn_smi = row[3] 
                all_rcts_smi, all_reag_smi, prod_smi = rxn_smi.split('>')
                all_rcts_smi, prod_smi = all_rcts_smi.split()[0], prod_smi.split()[0]
            
            rids = ','.join(sorted(re.findall(pt, all_rcts_smi)))
            pids = ','.join(sorted(re.findall(pt, prod_smi)))
            if rids != pids:  # atom mapping is not 1:1, but for rxn-ebm, this is not important
                bad_mapping += 1
                # continue

            all_rcts_mol = [Chem.MolFromSmiles(smi) for smi in all_rcts_smi.split('.')]

            mol_prod = Chem.MolFromSmiles(prod_smi)
            if mol_prod is None:  # rdkit is not able to parse the product, critical! 
                bad_prod += 1
                continue
                
            # check if all have atom mapping --> for rxn-ebm, this is not important
            if not all([a.HasProp('molAtomMapNumber') for a in mol_prod.GetAtoms()]):
                missing_map += 1
                # continue

            rxn_smi = move_reagents(
                            mol_prod, all_rcts_mol, all_reag_smi, keep_reagents, remove_rct_mapping)

            temp_rxn_smi = remove_mapping(rxn_smi, keep_reagents=keep_reagents)
            if len(temp_rxn_smi.split('>')[-1]) < 3: # check product
                too_small += 1 # e.g. 'Br', 'O', 'I'
                continue 
            if remove_all_mapping:
                rxn_smi = temp_rxn_smi
            clean_list.append(rxn_smi)  

            raw_num += 1
            # if raw_num % 10000 == 0:
            #     print(f'Extracted {raw_num} rxn_smis!')
        
        print('# clean rxn: ', len(clean_list))
        # print('# unique rxn:', len(set(clean_list))) 
        print('bad mapping: ', bad_mapping)
        print('bad prod: ', bad_prod)
        print('too small: ', too_small)
        print('missing map: ', missing_map)
        print('raw rxn extracted: ', raw_num, '\n')
        return clean_list

def clean_rxn_smis_all_datasets(raw_data_file_prefix: str, rxn_smi_file_prefix: str, 
                             root: Optional[str] = None, lines_to_skip: int = 1, 
                             keep_reagents: bool = False, remove_rct_mapping: bool = True,
                             distributed: bool = True):
    '''
    keep_reagents : bool (Default = True)
        whether to keep reagents in the output SMILES
    remove_rct_mapping : bool (Default = True)
        whether to remove atom mapping if atom in reactant is not in product (i.e. leaving groups)
    distributed : bool (Default = True)
        whether to distribute the computation across all possible workers 
    
    Also see: clean_rxn_smis_from_csv
    '''
    if root is None:
        root = Path(__file__).parents[2] 
    if distributed: 
        try:
            num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            num_workers = os.cpu_count()
        print(f'Parallelizing over {num_workers} cores')
    else:
        print(f'Not parallelizing!')
        num_workers = 1

    cleaned_rxn_smis = {'train': None, 'valid': None, 'test': None}
    for dataset in cleaned_rxn_smis.keys():
        print(f'Processing {dataset}')

        with Pool(max_workers=num_workers) as client:
            future = client.submit(
                clean_rxn_smis_from_csv, 
                root / 'data'/ 'original_data' / f'{raw_data_file_prefix}_{dataset}.csv', 
                lines_to_skip = lines_to_skip, keep_reagents=keep_reagents, remove_rct_mapping=remove_rct_mapping)  
            cleaned_rxn_smis[dataset] = future.result()

        with open(root / 'data' / 'cleaned_data' / f'{rxn_smi_file_prefix}_{dataset}.pickle', 'wb') as handle:
            pickle.dump(cleaned_rxn_smis[dataset], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def get_uniq_mol_smis_all_datasets(rxn_smi_file_prefix: str, mol_smis_filename: str, 
                                   root: Optional[str] = None):
    '''
    NOTE: does not collect reagents 
    '''
    if root is None:
        root = Path(__file__).parents[2] / 'data' / 'cleaned_data'

    # load cleaned_rxn_smis into a dictionary to be looped through 
    cleaned_rxn_smis = {'train': None, 'valid': None, 'test': None}
    for dataset in cleaned_rxn_smis.keys():
        with open(root / f'{rxn_smi_file_prefix}_{dataset}.pickle', 'rb') as handle:
            cleaned_rxn_smis[dataset] = pickle.load(handle)

    uniq_mol_smis = set()
    # loop through all 3 datasets, and collect all unique reactants & products (not reagents!)
    for dataset in cleaned_rxn_smis.keys(): 
        for rxn_smi in tqdm(cleaned_rxn_smis[dataset]):
            rcts = rxn_smi.split('>')[0]
            prod = rxn_smi.split('>')[-1]
            rcts_prod_smis = rcts + '.' + prod
            for mol_smi in rcts_prod_smis.split('.'):
                uniq_mol_smis.add(mol_smi)
    
    with open(root / mol_smis_filename, 'wb') as handle:
        pickle.dump(list(uniq_mol_smis), handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    raw_data_file_prefix = 'schneider50k_raw'  # user input
    rxn_smi_file_prefix = '50k_clean_rxnsmi_noreagent'  # user input

    clean_rxn_smis_all_datasets(raw_data_file_prefix, rxn_smi_file_prefix, lines_to_skip = 1, 
                             keep_reagents=False, remove_rct_mapping=True, distributed=True)
    
    mol_smis_filename = '50k_mol_smis.pickle' # user input
    get_uniq_mol_smis_all_datasets(rxn_smi_file_prefix, mol_smis_filename)

# textIO: the textIO object returned by 'open()' 