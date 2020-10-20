''' This module contains functions to clean a raw dataset of reaction SMILES strings, which includes
operations such as removing atom mapping and checking for invalid molecule SMILES. There is also
a function to generate a list of SMILES strings of all unique molecules in the given dataset. Note that
in rxn-ebm, a 'dataset' refers to the combination of 'train', 'valid', and 'test', (which are each called a 'split')
e.g. USPTO_50k is a dataset.
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
from typing import List, Optional, Union

Mol = rdkit.Chem.rdchem.Mol


def remove_mapping(rxn_smi: str, keep_reagents: bool = False) -> str:
    '''
    Removes all atom mapping from the reaction SMILES string

    Parameters
    ----------
    rxn_smi : str
        The reaction SMILES string whose atom mapping is to be removed
    keep_reagents : bool (Default = False)
        whether to keep the reagents in the output reaction SMILES string

    Returns
    -------
    str
        The reaction SMILES string with all atom mapping removed

    Also see: clean_rxn_smis_from_csv
    '''
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smi, useSmiles=True)
    if not keep_reagents:
        rxn.RemoveAgentTemplates()

    prods = [mol for mol in rxn.GetProducts()]
    for prod in prods:
        for atom in prod.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')

    rcts = [mol for mol in rxn.GetReactants()]
    for rct in rcts:
        for atom in rct.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')

    return rdChemReactions.ReactionToSmiles(rxn)


def move_reagents(
        mol_prod: Mol,
        reactants: List[Mol],
        original_reagents: List[str],
        keep_reagents: bool = False,
        remove_rct_mapping: bool = True) -> str:
    '''
    Adapted func from Hanjun Dai's GLN - gln/data_process/clean_uspto.py --> get_rxn_smiles()
    to additionally keep track of reagents (reactants that don't appear in products)
    Gets rid of reactants when they don't contribute to the product

    Parameters
    ----------
    mol_prod : Mol
        product molecule
    reactants : List[Mol]
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

    Also see: clean_rxn_smis_from_csv
    '''
    prod_smi = Chem.MolToSmiles(mol_prod, True)
    prod_maps = set(re.findall(r'\:([[0-9]+)\]', prod_smi))
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

                elif remove_rct_mapping:  # removes atom mapping if atom in reactant is not in product
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
    else:
        reagents_smi = ''

    return '{}>{}>{}'.format(reactants_smi, reagents_smi, prod_smi)


def clean_rxn_smis_from_csv(path_to_rxn_smis: Union[str,
                                                    bytes,
                                                    os.PathLike],
                            lines_to_skip: int = 1,
                            dataset_name: str = '50k',
                            split_mode: str = 'multi',
                            keep_reagents: bool = False,
                            remove_rct_mapping: bool = True,
                            remove_all_mapping: bool = True):
    '''
    Adapted function from Hanjun Dai's GLN: gln/data_process/clean_uspto.py --> main()
    NOTE: reads csv file twice, first time to get total line count for tqdm, second time to do the actual work
          This may not be practical with extremely large csv files

    Cleans reaction SMILES strings by removing those with:
        bad product (SMILES not parsable by rdkit)
        too small products, like 'O' (='H2O'), 'N'(='NH3'), i.e. a large reactant fails to be recorded as a product

    It also checks these, but does not remove them, since atom mapping is not critical for us:
        missing atom mapping (not all atoms in the product molecule have atom mapping),
        bad atom mapping (not 1:1 between reactants and products)

    Parameters
    ----------
    path_to_rxn_smis : str
        full path to the CSV file containing the reaction SMILES strings
        there will be one CSV file each for train, valid and test, coordinated by clean_rxn_smis_all_splits
    lines_to_skip : int (Default = 1)
        how many header lines to skip in the CSV file
        This is 1 for USPTO_50k (schneider), but 3 for USPTO_STEREO, and 1 for USPTO_FULL (GLN)
        Unfortunately, this cannot be reliably extracted from some automatic algorithm, as every CSV file can be differently formatted
    dataset_name : str (Default = '50k')
        the name of the dataset.
        currently accepts: '50k' for USPTO_50k, 'STEREO' for USPTO_STEREO (480k), 'FULL' for USPTO_FULL (1M)
    split_mode : str (Default = 'multi')
        whether to keep and process reaction SMILES containing multiple products, or ignore them
        Choose between 'single' and 'multi'
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
        list of cleaned reaction SMILES strings with possible duplicates
        NOTE: for USPTO_50k from schneider50k, only 4 reaction SMILES should be removed for having too small products
    set_clean_list : List[str]
        list of cleaned reaction SMILES strings without duplicates 
        this will be used if remove_dup_rxns is set to True in clean_rxn_smis_all_splits()
    bad_mapping_idxs : List[int]
        indices of reaction SMILES strings in original dataset with bad atom mapping (product atom id's do not all match reactant atom id's)
    bad_prod_idxs : List[int]
        indices of reaction SMILES strings in original dataset with bad products (not parsable by RDKit)
    too_small_idxs : List[int]
        indices of reaction SMILES strings in original dataset with too small products (product SMILES string smaller than 3 characters)
    missing_map_idxs : List[int]
        indices of reaction SMILES strings in original dataset with missing atom mapping
    dup_rxn_idxs : List[int]
        indices of reaction SMILES strings in original dataset that are duplicates of an already cleaned & extracted reaction SMILES string

    Also see: move_reagents, remove_mapping
    '''
    if remove_all_mapping:
        remove_rct_mapping = True

    pt = re.compile(r':(\d+)]')
    clean_list, set_clean_list = [], set()
    bad_mapping, bad_mapping_idxs = 0, []
    bad_prod, bad_prod_idxs = 0, []
    missing_map, missing_map_idxs = 0, []
    too_small, too_small_idxs = 0, []
    dup_rxn_idxs = [] 
    extracted = 0
    num_single, num_multi = 0, 0 # for USPTO_FULL, to keep track of #rxn with multiple products

    with open(path_to_rxn_smis, 'r') as csv_file:
        total_lines = len(csv_file.readlines()) - lines_to_skip

    with open(path_to_rxn_smis, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        for line in range(lines_to_skip):
            # skip first row of csv file (for USPTO_STEREO, 3 lines to be
            # skipped)
            header = next(reader)

        for i, row in enumerate(tqdm(reader, total=total_lines)):
            if dataset_name == '50k':
                rxn_smi = row[0].split(',')[2]  # second column of the csv file
                all_rcts_smi, all_reag_smi, prods_smi = rxn_smi.split('>')
            elif dataset_name == 'STEREO':
                rxn_smi = row[3]  # third column
                all_rcts_smi, all_reag_smi, prods_smi = rxn_smi.split('>')
                all_rcts_smi, prods_smi = all_rcts_smi.split()[0], prods_smi.split()[0]  # remove ' |f:1...'
            elif dataset_name == 'FULL':
                rxn_smi = row[header.index('ReactionSmiles')]
                all_rcts_smi, all_reag_smi, prods_smi = rxn_smi.split('>')
                all_rcts_smi = all_rcts_smi.split()[0]  # remove ' |f:1...'
                prods_smi = prods_smi.split()[0]  # remove ' |f:1...'
                if '.' in prods_smi:
                    num_multi += 1
                else:
                    num_single += 1
                if split_mode == 'single' and '.' in prods_smi:  # multiple prods
                    continue

            rids = ','.join(sorted(re.findall(pt, all_rcts_smi)))
            pids = ','.join(sorted(re.findall(pt, prods_smi)))
            if rids != pids:  # atom mapping is not 1:1, but for rxn-ebm, this is not important
                bad_mapping += 1
                bad_mapping_idxs.append(i)

            all_rcts_mol = [Chem.MolFromSmiles(
                smi) for smi in all_rcts_smi.split('.')]

            for prod_smi in prods_smi.split('.'):
                mol_prod = Chem.MolFromSmiles(prod_smi)
                if mol_prod is None:  # rdkit is not able to parse the product, critical!
                    bad_prod += 1
                    bad_prod_idxs.append(i)
                    continue

                # check if all have atom mapping, but for rxn-ebm, this is not
                # important
                if not all([a.HasProp('molAtomMapNumber')
                            for a in mol_prod.GetAtoms()]):
                    missing_map += 1
                    missing_map_idxs.append(i)

                rxn_smi = move_reagents(
                    mol_prod,
                    all_rcts_mol,
                    all_reag_smi,
                    keep_reagents,
                    remove_rct_mapping)

                temp_rxn_smi = remove_mapping(
                    rxn_smi, keep_reagents=keep_reagents)
                if len(temp_rxn_smi.split('>')[-1]) < 3:  # check length of product SMILES string
                    too_small += 1  # e.g. 'Br', 'O', 'I'
                    too_small_idxs.append(i)
                    continue
                if remove_all_mapping:
                    rxn_smi = temp_rxn_smi
                clean_list.append(rxn_smi)

                if rxn_smi not in set_clean_list:
                    set_clean_list.add(rxn_smi)
                else:
                    dup_rxn_idxs.append(i)

                extracted += 1

        print('# clean rxn: ', len(clean_list))
        print('# unique rxn:', len(set_clean_list))
        print('bad mapping: ', bad_mapping) 
        print('bad prod: ', bad_prod)  
        print('too small: ', too_small)  
        print('missing map: ', missing_map) 
        print('raw rxn extracted: ', extracted, '\n')
        return clean_list, list(set_clean_list), bad_mapping_idxs, bad_prod_idxs, too_small_idxs, missing_map_idxs, dup_rxn_idxs


def clean_rxn_smis_all_splits(input_file_prefix: str = 'schneider50k_raw',
                              output_file_prefix: str = '50k_clean_rxnsmi_noreagent',
                              input_root: Optional[Union[str,
                                                         bytes,
                                                         os.PathLike]] = None,
                              output_root: Optional[Union[str,
                                                          bytes,
                                                          os.PathLike]] = None,
                              dataset_name: str = '50k',
                              lines_to_skip: int = 1,
                              keep_reagents: bool = False,
                              remove_dup_rxns: bool = True,
                              remove_rct_mapping: bool = True,
                              remove_all_mapping: bool = True,
                              save_idxs: bool = True,
                              parallelize: bool = True):
    ''' NOTE: assumes file extension is .csv. GLN uses .rsmi for USPTO_FULL, but it's actually .csv file

    Parameters
    ----------
    input_file_prefix : str
        file prefix of the original raw, probably unclean, reaction SMILES csv file
        this function appends split = ['train', 'valid', 'test'] --> {raw_data_file_prefix}_{split}.csv
    output_file_prefix : str (Default = '50k_clean_rxnsmi_noreagent')
        file prefix of the output, cleaned reaction SMILES pickle file
        recommended format: '{size_of_dataset}_clean_rxnsmi_{any_tags}' example tags include 'noreagent', 'nostereo'
    input_root : Optional[Union[str, bytes, os.PathLike]] (Default = None)
        full file path to the folder containing the original raw data csv
        if None, we assume the original directory structure of rxn-ebm package, and set this to:
            path/to/rxnebm/data/original_data
    output_root : Optional[Union[str, bytes, os.PathLike]] (Default = None)
        full file path to the folder containing the output cleaned data pickle
        if None, we assume the original directory structure of rxn-ebm package, and set this to:
            path/to/rxnebm/data/cleaned_data
    dataset_name : str (Default = '50k')
        the name of the dataset.
        currently accepts: '50k' for USPTO_50k, 'STEREO' for USPTO_STEREO (480k), 'FULL' for USPTO_FULL (1M)
        This affects how the data is read from the raw file
    split_mode : str (Default = 'multi')
        whether to keep and process reaction SMILES containing multiple products, or ignore them
        Choose between 'single' and 'multi'
    lines_to_skip : int (Default = 1)
        how many header lines to skip in the CSV file
        This is 1 for USPTO_50k (schneider), but 3 for USPTO_STEREO, and 1 for USPTO_FULL (GLN)
        Unfortunately, this cannot be reliably extracted from some automatic algorithm, as every CSV file can be differently formatted
    keep_reagents : bool (Default = False)
        whether to keep reagents in the output SMILES string
    remove_dup_rxns : bool (Default = True)
        whether to remove duplicate rxn_smi
    remove_rct_mapping : bool (Default = True)
        whether to remove atom mapping if atom in reactant is not in product (i.e. leaving groups)
    remove_all_mapping : bool (Default = True)
        whether to remove all atom mapping from the reaction SMILES,
        if True, remove_rct_mapping will be automatically set to True
    save_idxs : bool (Default = True)
        whether to save all bad idxs to a file in the same directory as the output file 
    parallelize : bool (Default = True)
        whether to parallelize the computation across all possible workers

    Also see: clean_rxn_smis_from_csv
    '''
    if remove_all_mapping:
        remove_rct_mapping = True

    if input_root is None:
        input_root = Path(__file__).parents[2] / 'data' / 'original_data'
    if output_root is None:
        output_root = Path(__file__).parents[2] / 'data' / 'cleaned_data'

    if parallelize:
        try:
            num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            num_workers = os.cpu_count()
        print(f'Parallelizing over {num_workers} cores')
    else:
        print(f'Not parallelizing!')
        num_workers = 1

    cleaned_rxn_smis = {'train': None, 'valid': None, 'test': None}
    for split in cleaned_rxn_smis:
        print(f'Processing {split}')

        with Pool(max_workers=num_workers) as client:
            future = client.submit(
                clean_rxn_smis_from_csv,
                input_root / f'{input_file_prefix}_{split}.csv',
                dataset_name=dataset_name,
                lines_to_skip=lines_to_skip,
                keep_reagents=keep_reagents,
                remove_rct_mapping=remove_rct_mapping,
                remove_all_mapping=remove_all_mapping)
            if remove_dup_rxns:
                # [1] is set_clean_list
                cleaned_rxn_smis[split] = future.result()[1]
            else:
                cleaned_rxn_smis[split] = future.result()[0]
            
        if save_idxs:
            bad_idxs = {}
            bad_idxs['bad_mapping_idxs'] = future.result()[2]
            bad_idxs['bad_prod_idxs'] = future.result()[3]
            bad_idxs['too_small_idxs'] = future.result()[4]
            bad_idxs['missing_map_idxs'] = future.result()[5]
            bad_idxs['dup_rxn_idxs'] = future.result()[6]

            with open(output_root / f'{output_file_prefix}_{split}_badidxs.pickle', 'wb') as handle:
                pickle.dump(
                    bad_idxs, 
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_root / f'{output_file_prefix}_{split}.pickle', 'wb') as handle:
            pickle.dump(
                cleaned_rxn_smis[split],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)


def get_uniq_mol_smis_all_splits(rxn_smi_file_prefix: str = '50k_clean_rxnsmi_noreagent',
                                 output_filename: str = '50k_mol_smis',
                                 root: Optional[Union[str, bytes, os.PathLike]] = None):
    '''
    Parameters
    ----------
    rxn_smi_file_prefix : str (Default = '50k_clean_rxnsmi_noreagent')
        file prefix of the cleaned reaction SMILES pickle file
        set by output_file_prefix param in clean_rxn_smis_all_splits()
    output_filename : str (Default = '50k_mol_smis')
        filename of the output pickle file to contain the unique molecular SMILES
    root : Optional[Union[str, bytes, os.PathLike]] (Default = None)
        full path to the folder containing the cleaned rxn_smi_file & will contain the output file
        if None, we assume the original directory structure of rxn-ebm package, and set this to:
            path/to/rxnebm/data/cleaned_data

    NOTE: does not collect reagents
    '''
    if root is None:
        root = Path(__file__).parents[2] / 'data' / 'cleaned_data'
    if Path(output_filename).suffix != '.pickle':
        output_filename += '.pickle'

    # load cleaned_rxn_smis into a dictionary to be looped through
    cleaned_rxn_smis = {'train': None, 'valid': None, 'test': None}
    for split in cleaned_rxn_smis:
        with open(root / f'{rxn_smi_file_prefix}_{split}.pickle', 'rb') as handle:
            cleaned_rxn_smis[split] = pickle.load(handle)

    uniq_mol_smis = set()
    # loop through all 3 splits, and collect all unique reactants & products
    # (not reagents!)
    for split in cleaned_rxn_smis:
        for rxn_smi in tqdm(cleaned_rxn_smis[split]):
            rcts = rxn_smi.split('>')[0]
            prod = rxn_smi.split('>')[-1]
            rcts_prod_smis = rcts + '.' + prod
            for mol_smi in rcts_prod_smis.split('.'):
                uniq_mol_smis.add(mol_smi)

    with open(root / output_filename, 'wb') as handle:
        pickle.dump(
            list(uniq_mol_smis),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__': 
    clean_rxn_smis_all_splits(
        'USPTO_FULL_GLN',
        'USPTO_FULL_GLN_clean_rxnsmi_noreagent',
        dataset_name='FULL',
        lines_to_skip=1)
    get_uniq_mol_smis_all_splits(rxn_smi_file_prefix='USPTO_FULL_GLN_clean_rxnsmi_noreagent',
                                 output_filename='USPTO_FULL_mol_smis')

    # to clean USPTO_50k
    # clean_rxn_smis_all_splits(
    #     'schneider50k_raw',
    #     '50k_clean_rxnsmi_noreagent',
    #     dataset_name='50k',
    #     lines_to_skip=1)
    # get_uniq_mol_smis_all_splits(rxn_smi_file_prefix='50k_clean_rxnsmi_noreagent',
                                #  output_filename='50k_mol_smis')


# code from clean_uspto.py of GLN for splitting USPTO_FULL <AFTER> cleaning into 80/10/10
#     seed = 19260817
#     np.random.seed(seed)
#     random.seed(seed)
'''
    random.shuffle(clean_list)

    num_val = num_test = int(len(clean_list) * 0.1)

    out_folder = '.'
    for phase in ['val', 'test', 'train']:
        fout = os.path.join(out_folder, 'raw_%s.csv' % phase)
        with open(fout, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'reactants>reagents>production'])

            if phase == 'val':
                r = range(num_val)
            elif phase == 'test':
                r = range(num_val, num_val + num_test)
            else:
                r = range(num_val + num_test, len(clean_list))
            for i in r:
                rxn_smiles = clean_list[i][1].split('>')
                result = []
                for r in rxn_smiles:
                    if len(r.strip()):
                        r = r.split()[0]
                    result.append(r)
                rxn_smiles = '>'.join(result)
                writer.writerow([clean_list[i][0], rxn_smiles])
'''
