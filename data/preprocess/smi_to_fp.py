''' This module contains functions to process an already cleaned list of reaction SMILES strings
into molecular fingerprints and reaction fingerprints, which are then used as inputs into the 
energy-based models, as well as to prepare negative reaction examples
'''

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator, GetMorganGenerator

import scipy 
from scipy import sparse
import numpy as np

import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

def gen_lookup_dict_from_file(mol_smis_filename: Union[str, bytes, os.PathLike]='50k_mol_smis.pickle', 
                        output_filename: Union[str, bytes, os.PathLike]='50k_mol_smi_to_sparse_fp_idx.pickle', 
                        bad_smis_filename: Optional[Union[str, bytes, os.PathLike]]=None, 
                        root: Optional[Union[str, bytes, os.PathLike]] = None) -> dict: 
    ''' 
    Generates a lookup dictionary mapping each molecular SMILES string in the given
    mol_smis .pickle file to its index. ENSURE that the order of elements in
    the mol_smis .pickle file (and ALL its derivatives, like mol_fp .npz file) are ALL identical.
    Any shuffling will render this dictionary useless!  
    '''
    if root is None: # if not provided, goes up 2 levels to get to 'rxn-ebm/'
        root = Path(__file__).parents[2] / 'data' / 'cleaned_data'   
    if Path(output_filename).suffix != '.pickle': 
        output_filename += '.pickle'
    if (root / output_filename).exists():
        print(f'At: {root / output_filename}')
        print('The lookup_dict file already exists!')
        return 
    if Path(mol_smis_filename).suffix != '.pickle':
        mol_smis_filename += '.pickle'
    with open(root / mol_smis_filename, 'rb') as handle:
        mol_smis = pickle.load(handle)

    mol_smi_to_fp = {}
    for i, mol_smi in enumerate(tqdm(mol_smis)):
        mol_smi_to_fp[mol_smi] = i

    # gen_count_mol_fps_from_file --> returns bad_smis, if not empty list --> provide bad_smis_filename
    if bad_smis_filename:
        if Path(bad_smis_filename).suffix != '.pickle':
            bad_smis_filename += '.pickle'
        with open(root / bad_smis_filename, 'rb') as handle:
            bad_smis = pickle.load(handle)
        for bad_smi in bad_smis:
            mol_smi_to_fp[bad_smi] = -1 # bad_smi's all point to index -1       
        
    with open(root / output_filename, 'wb') as handle:
        pickle.dump(mol_smi_to_fp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return mol_smi_to_fp

def mol_smi_to_count_fp(mol_smi: str, radius: int = 3, fp_size: int = 4096, 
                        dtype: str = 'int16') -> scipy.sparse.csr_matrix :
    fp_gen = GetMorganGenerator(radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size)
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return count_fp

# def mol_smi_to_count_fp(mol_smi: str, radius: int = 3, fp_size: int = 4096, 
#                         dtype: str = 'int16') -> scipy.sparse.csr_matrix :
#     fp_gen = GetRDKitFPGenerator(minPath=1, maxPath=1+radius, fpSize=fp_size)
#     mol = Chem.MolFromSmiles(mol_smi)
#     uint_count_fp = fp_gen.GetCountFingerprint(mol)
#     count_fp = np.empty((1, fp_size), dtype=dtype)
#     DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
#     return count_fp

def gen_count_mol_fps_from_file(mol_smis_filename: Union[str, bytes, os.PathLike] = '50k_mol_smis.pickle', 
                            output_filename: Union[str, bytes, os.PathLike] = '50k_count_mol_fps.npz', 
                            radius: int = 3, fp_size: int = 4096, dtype: str = 'int16',
                            root: Optional[Union[str, bytes, os.PathLike]] = None) -> Tuple[Path, Path]:
    ''' TODO: add docstring
    '''
    if root is None:  
        root = Path(__file__).parents[2] / 'data' / 'cleaned_data' 
    if Path(output_filename).suffix != '.npz': 
        output_filename += '.npz'
    cleaned_mol_smis_path = root / str(Path(mol_smis_filename).stem + '_count.pickle')
    bad_mol_smis_path = root / str(Path(mol_smis_filename).stem + '_bad.pickle') 
    if (root / output_filename).exists():
        print(f'At: {root / output_filename}')
        print('The count_mol_fp file already exists!') 
    if Path(mol_smis_filename).suffix != '.pickle': 
        mol_smis_filename += '.pickle'
    with open(root / mol_smis_filename, 'rb') as handle:
        mol_smis = pickle.load(handle)

    count_mol_fps, bad_idx = [], []
    for i, mol_smi in enumerate(tqdm(mol_smis, total=len(mol_smis))):
        cand_mol_fp = mol_smi_to_count_fp(mol_smi, radius, fp_size, dtype)
        if sum(cand_mol_fp) == 0: # candidate mol_fp is all 0's 
            bad_idx.append(i)
            continue # don't add, and keep track of index (or replace with our favorite imputation, like adding +1 at 0'th index)
        else: # sparsify then append to list
            count_mol_fps.append(sparse.csr_matrix(cand_mol_fp, dtype=dtype))

    bad_smis = []
    if bad_idx: # to catch small molecules like 'O', 'Cl', 'N' that give count vector of all 0 values
        count_mol_fps.append(sparse.csr_matrix((1, fp_size), dtype=dtype))
        idx_offset = 0 # offset is needed as the indices will dynamically change after popping each bad_smi
        for idx in bad_idx:
            bad_smis.append(mol_smis[idx - idx_offset])
            mol_smis.pop(idx - idx_offset)
            idx_offset += 1
        print(f'The bad SMILES are {bad_smis}')
        print(f'Saving bad_mol_smis at {bad_mol_smis_path}')
        with open(bad_mol_smis_path, 'wb') as handle:
            pickle.dump(bad_smis, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        print(f'Resaving mol_smis as {cleaned_mol_smis_path}')
        with open(cleaned_mol_smis_path, 'wb') as handle:
            pickle.dump(mol_smis, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    else: # bad_smis = []
        print('No bad SMILES detected!') 

    count_mol_fps = sparse.vstack(count_mol_fps)
    sparse.save_npz(root / output_filename, count_mol_fps)

def mol_smi_to_bit_fp(mol_smi: str, radius: int = 3, fp_size: int = 4096, 
                     dtype: str = 'int8') -> scipy.sparse.csr_matrix :
    mol = Chem.MolFromSmiles(mol_smi)
    bitvect_fp = AllChem.GetMorganFingerprintAsBitVect( 
                        mol=mol, radius=radius, nBits=fp_size, useChirality=True)
    bit_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(bitvect_fp, bit_fp)
    return bit_fp

def gen_bit_mol_fps_from_file(mol_smis_filename: Union[str, bytes, os.PathLike], 
                            output_filename: Union[str, bytes, os.PathLike], 
                            radius: int = 3, fp_size: int = 4096, dtype: str = 'int8',
                            root: Optional[Union[str, bytes, os.PathLike]] = None) -> Path:
    '''
    Returns
    -------
    Path : 
        pathlib's Path object to the output bit mol_fps file  
    '''
    if root is None:  
        root = Path(__file__).parents[2] / 'data' / 'cleaned_data' 
    if Path(output_filename).suffix != '.npz': 
        output_filename += '.npz'
    if (root / output_filename).exists():
        print(f'At: {root / output_filename}')
        print('The bit_mol_fp file already exists!')
        return root / output_filename

    if Path(mol_smis_filename).suffix != '.pickle': 
        mol_smis_filename += '.pickle'
    with open(root / mol_smis_filename, 'rb') as handle:
        mol_smis = pickle.load(handle)

    bit_mol_fps = [] 
    for mol_smi in tqdm(mol_smis, total=len(mol_smis)):
        bit_fp = mol_smi_to_bit_fp(mol_smi, radius, fp_size, dtype)
        bit_mol_fps.append(sparse.csr_matrix(bit_fp, dtype=dtype))
    bit_mol_fps = sparse.vstack(bit_mol_fps)
    sparse.save_npz(root / output_filename, bit_mol_fps)
    return root / output_filename

def rxn_smi_to_raw_fp(rxn_smi: str, lookup_dict, dict, 
                      sparse_mol_fps: scipy.sparse.csr_matrix, max_rcts: int) -> Tuple[scipy.sparse.csr_matrix, int]:
    ''' TODO: make list_rxn_smis_to_raw_fps  function
    Given a reaction SMILES string, splits it into reactants and product, and then
    uses the lookup dict to index into the correct row of the sparse fp matrix, 
    and concatenates molecular fingerprints into a long, row sparse vector, along with 
    information on number of reactants in each reaction

    Agnostic to the type of fingerprint used, as long as they are in a scipy.sparse.csr_matrix format

    Parameters
    ----------
    rxn_smi : str
        The reaction SMILES string to be converted    
    lookup_dict : dict
        The lookup dict mapping each unique molecule SMILES string to the corresponding index in the 
        sparse matrix of molecular fingerprints
    sparse_mol_fps : scipy.sparse.csr_matrix
        The sparse matrix containing the molecular fingerprints. Accepts both bit and count fingerprints
    max_rcts : int
        The maximum number of reactants possible in any reaction for the given dataset (train + valid + test)

    Returns
    -------
    raw_fp : scipy.sparse.csr_matrix
        Each row contains horizontally stacked sparse fingerprints of each reactant molecule 
        followed by the product molecule, i.e. each row is: 1 x (reactant fp size x max #reactants in entire dataset) 
    num_rcts : int
        An integer stating the number of reactants in that reaction
    '''
    # from benchmarks, csr_matrix is almost 7x faster than lil_matrix! 
    rct_fps = sparse.csr_matrix((max_rcts, sparse_mol_fps[0].shape[1]), dtype = sparse_mol_fps[0].dtype) 
    # prod_fp = sparse.csr_matrix((1, sparse_mol_fps[0].shape[1]), dtype = sparse_mol_fps[0].dtype)

    rcts_smi = rxn_smi.split('>')[0].split('.') 
    for i, rct_smi in enumerate(rcts_smi):
        try:
            rct_index = lookup_dict[rct_smi]
            rct_fps[i] = sparse_mol_fps[rct_index]
        except KeyError: # mol_smi is not in lookup_dict
            rct_fps[i] = sparse.csr_matrix(np.zeros((1, sparse_mol_fps[0].shape[1])), dtype = sparse_mol_fps[0].dtype)
#         print(len(rct_fps[i].nonzero()[0])) # these have to sum up to len(raw_fp.nonzero()[0]) 
    rct_fps = rct_fps.reshape((1, -1))

    prod_smi = rxn_smi.split('>')[-1]
    prod_index = lookup_dict[prod_smi]
    prod_fp = sparse_mol_fps[prod_index]
#     print(len(prod_fp[0].nonzero()[0])) # these have to sum up to len(raw_fp.nonzero()[0]) 

    raw_fp = sparse.hstack([rct_fps, prod_fp]).tocsr()
    return raw_fp, len(rcts_smi)

def list_rxn_smis_to_raw_fps(rxn_smis_dataset: List[str], lookup_dict: dict,
                             sparse_mol_fps: scipy.sparse.csr_matrix, max_rcts: int) -> scipy.sparse.csr_matrix:
    '''
    Iterates through a list of reaction SMILES strings and uses rxn_smi_to_raw_fp to 
    convert them into a tuple: (sparse matrix of raw molecular fingerprints, list of the number of reactants in each reaction)
    '''
    list_sparse_raw_fps, list_num_rcts = [], []
    for rxn_smi in tqdm(rxn_smis_dataset, total=len(rxn_smis_dataset)):
        raw_fp, num_rcts = rxn_smi_to_raw_fp(rxn_smi, lookup_dict, sparse_mol_fps, max_rcts)
        list_sparse_raw_fps.append(raw_fp)
        list_num_rcts.append(num_rcts)
    
    raw_fps_matrix = sparse.vstack(list_sparse_raw_fps)
    return raw_fps_matrix, list_num_rcts

def gen_raw_fps_from_file():
    ''' 
    TODO: realised that this function is almost identical to gen_diff_fps_from_file
    just that this uses list_rxn_smis_to_raw_fps, instead of list_rxn_smis_to_diff_fps 

    Therefore, to make this suite of functions more intuitive, maybe it's better for me to 
    wrap all of them in a class, maybe called smiEncoder, which, at __init__, loads the needed 
    files (or makes them, if they don't yet exist, like the lookup_dict, the mol_fp files)
    and then the user can call whatever functions they want (like gen_raw_fps or gen_diff_fps)
    The user will have to provide some parameters, like the filenames/paths, root path, 
    the fp_type (count or bit, for now), fp_size, radius, dtype, output filename, whether to do
    distributed processing
    '''
    return

def rxn_smi_to_diff_fp(rxn_smi: str, lookup_dict: dict, 
                      sparse_mol_fps: scipy.sparse.csr_matrix, fp_type: str = 'count') -> scipy.sparse.csr_matrix:
    '''
    Given a reaction SMILES string, splits it into reactants and product, and then
    uses the lookup dict to access the correct row of the sparse molecular fp matrix, 
    and then builds the difference fingerprint
    
    Accepts both count and bit fingerprints.
    For bit fingerprints, adds an additional step of capping the maximum and minimum values 
    to be 1 and -1 respectively 
    
    Parameters
    ----------
    rxn_smi : str
        The reaction SMILES string to be converted    
    lookup_dict : dict
        The lookup dict mapping each unique molecule SMILES string to the corresponding index in the 
        sparse matrix of molecular fingerprints
    sparse_mol_fps : scipy.sparse.csr_matrix
        The sparse matrix containing the molecular fingerprints
    fp_type : str (Default = 'count')
        The fingerprint type. Affects the maximum allowed value of the resulting difference fingerprint
        For a bit fingerprint, this will be 1. For a count fingerprint, this will be integers > 1. 
        
    Inferred/Automatic parameters
    -------------------
    These are automatically determined 
    fp_size : int (Default = 4096)
        The length of each input molecular fingerprint, 
        which will also be the length of the output sparse matrix
    dtype : str (Default = 'int16')
        The data type of the output sparse matrix
        Recommended: 'int16' for count vectors, 'int8' for bit vectors
        For bit vectors, we store them as 'boolean' arrays first when summing reactant fingerprints
        This prevents bit values from exceeding 1. Finally we convert to 'int8' to substract from the product fingerprint
    
    Returns
    -------
    diff_fp : scipy.sparse.csr_matrix
        The difference fingerprint which is in sparse matrix format with the same shape as the 
        original molecular fingerprint 
        
    Also see: rxn_smi_to_diff_fp, which calls this function
    '''
    if fp_type == 'bit': 
        dtype = 'bool' # must use boolean array to not exceed bit value of 1 when summing
    else:
        dtype = sparse_mol_fps[0].dtype  # infer dtype, for count fingerprints

    rcts_smi = rxn_smi.split('>')[0].split('.')
    diff_fp = sparse.csr_matrix(sparse_mol_fps[0].shape, dtype=dtype) # infer fp_size
    for rct_smi in rcts_smi:
        try:
            rct_index = lookup_dict[rct_smi]
            sparse_rct_fp = sparse_mol_fps[rct_index]
            diff_fp = diff_fp + sparse_rct_fp
        except KeyError: # one of the small molecules giving bad SMILES, which give completely 0 count fingerprint
            continue
 
    if fp_type == 'bit':
        diff_fp = diff_fp.astype('int8') # convert to 'int8' for final subtraction
    prod_smi = rxn_smi.split('>')[-1]
    prod_index = lookup_dict[prod_smi]
    sparse_prod_fp = sparse_mol_fps[prod_index]
    diff_fp = sparse_prod_fp - diff_fp
    return diff_fp

def list_rxn_smis_to_diff_fps(rxn_smis_dataset: List[str], lookup_dict: dict,
                             sparse_mol_fps: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    ''' 
    Iterates through given list of reaction SMILES strings and uses rxn_smi_to_diff_fp to 
    generate difference fingerprints, given the lookup dict to access the corresponding sparse molecular fingerprints
    
    Parameters
    ----------
    rxn_smi_dataset : List[str]
        A list of reaction SMILES strings
        There should be a separate dataset for train, valid & test
    lookup_dict : dict
        The lookup dict mapping each unique molecule SMILES string to the corresponding index in the 
        sparse matrix of molecular fingerprints
    sparse_mol_fps : scipy.sparse.csr_matrix
        The sparse matrix containing the molecular fingerprints
        
    Returns
    -------
    diff_fps_matrix : scipy.sparse.csr_matrix 
        A sparse matrix where each row is one difference reaction fingerprint
    
    Also see: rxn_smi_to_diff_fp
    '''        
    # iterate through rxn_smis and use mol_fp_to_diff_fp to do the conversion
    list_sparse_diff_fps = []
    for rxn_smi in tqdm(rxn_smis_dataset, total=len(rxn_smis_dataset)):
        diff_fp = rxn_smi_to_diff_fp(rxn_smi, lookup_dict, sparse_mol_fps)
        list_sparse_diff_fps.append(diff_fp)
    
    diff_fps_matrix = sparse.vstack(list_sparse_diff_fps)
    return diff_fps_matrix

def gen_diff_fps_from_file(rxn_smi_file_prefix: str='50k_clean_rxnsmi_noreagent', 
                        lookup_dict_filename: Union[str, bytes, os.PathLike]='50k_mol_smi_to_sparse_fp_idx.pickle', 
                        sparse_mol_fps_filename: Union[str, bytes, os.PathLike]='50k_count_mol_fps.npz', 
                        output_matrix_file_prefix: str='50k_count_diff_fps', 
                        root: Optional[Union[str, bytes, os.PathLike]] = None, distributed: bool = False):
    '''
    Generates difference fingerprints from reaction SMILES .pickle files for a whole dataset
    (train, valid and test) after also loading the corresponding lookup dict and sparse molecular fingerprint matrix
    
    Parameters
    ----------
    rxn_smi_file_prefix : str
        Filename prefix to the reaction SMILES .pickle file
        Internally, we append f'_{train/valid/test}.pickle' to this prefix to get the rxn smi's filename
    lookup_dict_filename : Union[str, bytes, os.PathLike]
        Filename of the lookup dict .pickle file; if '.pickle' is not included, automatically adds it
        There will only be one lookup dict file for one whole dataset 
        (e.g. USPTO_50k has one, USPTO_STEREO has one)
    sparse_mol_fps_filename : Union[str, bytes, os.PathLike]
        Filename of the pre-computed sparse molecular fingerprints .npz file; if '.npz' is not included, automatically adds it
        Just like the lookup dict, there will only be one sparse molecular fingerprint matrix for one whole dataset
        Accepts both count and bit morgan fingerprints
    output_matrix_file_prefix : str
        Filename prefix to save the output sparse matrix 
        Internally, we append f'_{train/valid/test}.pickle' to this prefix to get the output filename
    root : Optional[Union[str, bytes, os.PathLike]] (Default = None)
        The root directory from which to load and save files 
        This script, smi_to_fp.py should be located in rxn-ebm/data/preprocess/smi_to_fp.py;
        by default, we assign root to: Path(__file__).parents[2] / 'data' / 'cleaned_data' 
    distributed : bool (Default = False)
        whether to distribute the computation across all possible workers 
    
    Saves
    -----
    diff_fps_matrix : scipy.sparse.csr_matrix
        Saves the output sparse matrix of difference fingerprints using output_matrix_filename
        
    Also see: list_rxn_smis_to_diff_fps, rxn_smi_to_diff_fp
    '''    
    if root is None: # if not provided, goes up 2 levels to get to rxn-ebm/ then add data/cleaned_data/
        root = Path(__file__).parents[2] / 'data' / 'cleaned_data' 
    if (root / f'{output_matrix_file_prefix}_train.npz').exists():
        print(f'At: {root / f"{output_matrix_file_prefix}_train.npz"}')
        print('The sparse diff_fp file already exists!')
        print('Please check the directory again.')
        return 

    # LOAD LOOKUP_DICT & SPARSE_MOL_FPS (do not need to reload for train, valid and test)
    if Path(lookup_dict_filename).suffix != '.pickle':  
        lookup_dict_filename += '.pickle'
    if Path(sparse_mol_fps_filename).suffix != '.npz': 
        sparse_mol_fps_filename += '.npz'
    
    sparse_mol_fps = sparse.load_npz(root / sparse_mol_fps_filename)
    with open(root / lookup_dict_filename, 'rb') as handle:
        lookup_dict = pickle.load(handle)

    datasets = ['train', 'valid', 'test']
    for dataset in datasets:
        print(f'Processing {dataset}...')
        time.sleep(0.5) # to give time for printing before tqdm progress bar appears
 
        with open(root / f'{rxn_smi_file_prefix}_{dataset}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)
        if distributed: 
            try:
                num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                num_workers = os.cpu_count()
            print(f'Parallelizing over {num_workers} cores')
        else:
            print('Not parallelizing!')
            num_workers = 1

        with Pool(max_workers=num_workers) as client:
            future = client.submit(list_rxn_smis_to_diff_fps, rxn_smis, lookup_dict, sparse_mol_fps)
            diff_fps_matrix = future.result()
        sparse.save_npz(root / f'{output_matrix_file_prefix}_{dataset}.npz' , diff_fps_matrix)

if __name__ == '__main__':   
    gen_count_mol_fps_from_file()
    gen_lookup_dict_from_file()
    gen_diff_fps_from_file()