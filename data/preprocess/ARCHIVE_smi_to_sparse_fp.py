import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from tqdm import tqdm

import scipy 
from scipy import sparse

from typing import List, Optional
    
def rxn_smi_to_mol_fps(rxn_smi: str, fp_type: str = 'diff', 
                        radius: int = 3, rctfp_size: int = 4096, prodfp_size: int = 4096, 
                        max_rcts: int = 4, 
                        useChirality: bool = True, dtype: str = 'int8'):
    '''
    Parameters
    ----------
    rxn_smi : str
        the reaction SMILES to be converted into individual molecular fingerprints 
    fp_type : str (default = 'diff')
        'precomp' (precomputation):
            Creates raw MorganFP for each reactant & product molecule to be augmented/modified during training
            Also returns an integer #reactants for each reaction 

        'diff' (difference):
            Creates reaction MorganFP following Schneider et al in J. Chem. Inf. Model. 2015, 55, 1, 39–53
            reactionFP = productFP - sum(reactantFPs)
        
        'sep' (separate):
            Creates separate reactantsFP and productFP following Gao et al in ACS Cent. Sci. 2018, 4, 11, 1465–1476
    radius : int (default = 3)
        fingerprint radius (suggested: 2 or 3)
    rctfp_size : int (default = 4096)
        reactant fingerprint length (suggested: 2048 or 4096)
    prodfp_size : int (default = 4096)
        product fingerprint length (suggested: equal to rctfp_size)
    max_rcts : int
        maximum number of reactants for a reaction in the given rxn_smi_dataset (see rxn_smi_to_sparse_fps)
        e.g. USPTO_50k has 4 maximum reactants in any reaction
    fp_type : str (default = 'precomp')
        mode of fingerprint calculation, see raw_fp_from_smi for details
    dtype : str (default = 'bool')
        datatype to store fingerprints
    useChirality : bool (default = True)
        whether to use chirality information in fingerprint calculation
    
    '''
    # initialise numpy arrays to store fingerprints
    if fp_type == 'diff':
        assert rctfp_size == prodfp_size, 'rctfp_size != prodfp_size, unable to build diff fingerprint!!!'
        diff_fp = np.zeros(rctfp_size, dtype = dtype)
    elif fp_type == 'sep':
        rcts_fp = np.zeros(rctfp_size, dtype = dtype)
        prod_fp = np.zeros(prodfp_size, dtype = dtype)
    elif fp_type == 'precomp':
        assert rctfp_size == prodfp_size, 'rctfp_size != prodfp_size, unable to build sparse matrix!!!'
        rct_fps = np.zeros((max_rcts, rctfp_size), dtype = dtype) 
        prod_fp = np.zeros((1, rctfp_size), dtype = dtype)
    else:
        print('ERROR: fp_type not recognised!')
        return
    
    # create product FP
    prod_mol = Chem.MolFromSmiles(rxn_smi.split('>')[-1])
    try:
        prod_fp_bit = AllChem.GetMorganFingerprintAsBitVect( 
                        mol=prod_mol, radius=radius, nBits=prodfp_size, useChirality=useChirality)
        if fp_type == 'precomp':
            DataStructs.ConvertToNumpyArray(prod_fp_bit, prod_fp[0, :])
        else:      # on-the-fly creation of MorganFP during training  
            fp = np.zeros(prodfp_size, dtype = dtype)   # temporarily store numpy array as fp 
            DataStructs.ConvertToNumpyArray(prod_fp_bit, fp)
            if fp_type == 'diff':
                diff_fp = fp
            elif fp_type == 'sep':
                prod_fp = fp
    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return
                                  
    # create reactant FPs, subtracting each from product FP
    rcts_smi = rxn_smi.split('>')[0].split('.')
    for i, rct_smi in enumerate(rcts_smi):
        rct_mol = Chem.MolFromSmiles(rct_smi)
        try:
            rct_fp_bit = AllChem.GetMorganFingerprintAsBitVect(
                            mol=rct_mol, radius=radius, nBits=rctfp_size, useChirality=useChirality)
            if fp_type == 'precomp':
                DataStructs.ConvertToNumpyArray(rct_fp_bit, rct_fps[i, :])
            else:     # on-the-fly creation of MorganFP during training  
                fp = np.zeros(rctfp_size, dtype = dtype)
                DataStructs.ConvertToNumpyArray(rct_fp_bit, fp)
                if fp_type == 'diff':
                    diff_fp -= fp
                elif fp_type == 'sep':
                    rcts_fp += fp
        except Exception as e:
            print("Cannot build reactant fp due to {}".format(e))
            return
    
    if fp_type == 'diff':
        return diff_fp
    elif fp_type == 'sep':
        return np.concatenate([rcts_fp, prod_fp])
    elif fp_type == 'precomp':
        rct_fps = rct_fps.reshape(1, -1) # flatten into 1 long row-array 
        # return np.concatenate([rct_fps, prod_fp, np.array([[len(rcts_smi)]])], axis=1)
        return np.concatenate([rct_fps, prod_fp], axis=1, dtype='int8'), len(rcts_smi)
    
def rxn_smi_to_sparse_fps(rxn_smi_dataset: List[str], 
                   radius: int, fp_size: int, max_rcts: int, 
                   fp_type: str = 'precomp', dtype:str = 'int8',
                   useChirality: bool = True) -> scipy.sparse.csr_matrix :
    '''
    Parameters
    ----------
    rxn_smi_dataset : List[str]
        list of reaction SMILES
    radius : int
        fingerprint radius (suggested: 2 or 3)
    fp_size : int
        fingerprint length (suggested: 2048 or 4096)
    max_rcts : int
        maximum number of reactants for a reaction in the given rxn_smi_dataset
        e.g. USPTO_50k has 4 maximum reactants in any reaction
    fp_type : str (default = 'precomp')
        mode of fingerprint calculation, see raw_fp_from_smi for details
    dtype : str (default = 'bool')
        datatype to store fingerprints
    useChirality : bool (default = True)
        whether to use chirality information in fingerprint calculation

    Returns
    -------
    sparse_fps : scipy.sparse.csr_matrix
        a matrix of fingerprints
        if fp_type == 'precomp':
            has #reactants (len(rcts_smi)) for each rxn appended to each row 

    Also see: rxn_smi_to_mol_fps
    '''
    sparse_fps, list_num_rcts = [], []   
    for rxn_smi in tqdm(rxn_smi_dataset):          
        rxn_fp, num_rcts = raw_fp_from_smi(rxn_smi, fp_type=fp_type, 
                            radius=radius, rctfp_size=fp_size, prodfp_size=fp_size, 
                            max_rcts=max_rcts, 
                            useChirality=useChirality, dtype=dtype)
        rxn_fp_sparse = sparse.csr_matrix(rxn_fp, dtype=dtype)
        sparse_fps.append(rxn_fp_sparse)
        list_num_rcts.append(num_rcts)

    sparse_fps = sparse.vstack(sparse_fps, dtype=dtype)
    return sparse_fps, list_num_rcts

def gen_sparse_fps_from_uniq_mols():
    ''' 
    TODO: add arguments and docstring 
    '''
    root = Path(__file__).parents[1]
    uniq_mol_smi_output_prefix = '50k_unique_rcts_prod_smis.pickle'

    with open(root / 'cleaned_data' / uniq_mol_smi_output_prefix, 'rb') as handle:
        mol_smis = pickle.load(handle)

    mol_FPs_sparse = []  
    for mol_smi in tqdm(mol_smis):
        mol = Chem.MolFromSmiles(mol_smi)
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(
                mol=mol, radius=3, nBits=4096, useChirality=True)
        mol_FP = np.empty((1, 4096), dtype='int8')
        DataStructs.ConvertToNumpyArray(fp_bit, mol_FP)
        mol_FPs_sparse.append(sparse.csr_matrix(mol_FP, dtype='int8'))

    mol_FPs_sparse = sparse.vstack(mol_FPs_sparse, dtype='int8') 
    output_filename = '50k_all_mols_sparse_FPs.npz'
    sparse.save_npz(root / 'cleaned_data' / output_filename, mol_FPs_sparse)

    print('successfully built and saved molecule fingerprints!')

def gen_sparse_fps_from_file(filepath: str, path_to_sparse: Optional[str] = None, 
                            name: Optional[str] = None):
    ''' 
    Parses a .pickle file containing cleaned, reaction SMILES strings to generate a .npz file 
    containing sparse fingerprints for each molecule involved in the reaction, and 
    a .pickle file containing a list of the corresponding number of reactants in each reaction

    sparse_fps_with_numrcts: scipy.sparse.csr_matrix
        each row contains horizontally stacked sparse fingerprints of each reactant molecule (boolean) 
        followed by the product molecule, and lastly an integer stating the number of reactants in that reaction
        i.e. each row is of size 1 x (reactant fingerprint size x max #reactants in entire dataset + 1)

    Parameters
    ----------
    filepath : str
        the filepath of the pickle file containing the cleaned, reaction SMILES strings
    path_to_sparse : Optional[str] (Default = None)
        the path under which the .npz file should be written
    name_sparse : Optional[str] (Default = None)
        the name of the output .npz file
    name_list : Optional[str] (Default = None)
        the name of the output .pickle file containing a list of the number of reactants in each reaction,
        with the row order corresponding to the output .npz file 

    Returns
    ------
    # sparse_name : str
    #     the filename of the .npz file containing the sparse molecular fingerprints 

    Also see: rxn_smi_to_sparse_fps, rxn_smi_to_mol_fps
    '''
    import pickle
    from scipy import sparse
    from concurrent.futures import ProcessPoolExecutor as Pool
    import os 
    from pathlib import Path

    # path is relative to data folder
    path_to_SMILES_pkl = 'cleaned_data/clean_rxn_50k_nomap_noreagent.pickle' # to parse

    # path_to_sparse will be appended by '_{key}.npz' where key is 'train', 'valid' and 'test'
    path_to_sparse = 'cleaned_data/clean_rxn_50k_sparse_FPs_numrcts' 
    # path_to_sparse = 'USPTO_50k_data/clean_rxn_50k_sparse_rxnFPs' # for difference reaction fingerprints

    try:
        num_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        num_workers = os.cpu_count()
    print(f'Parallelizing over {num_workers} cores')

    with open(Path(__file__).parents[1] / path_to_SMILES_pkl , 'rb') as handle:
        clean_rxn_smi = pickle.load(handle)

    for key in ['train', 'valid', 'test']:
        print(f'\nMaking sparse FPs for {key}')
        with Pool(max_workers=num_workers) as client:
            future = client.submit(sparse_from_fp, clean_rxn_smi[key], 3, 4096, 4, 'precomp', 'bool') # to parse
            sparse_FPs_numrcts = future.result()
        sparse_name = path_to_sparse + f'_{key}.npz'
        sparse.save_npz(Path(__file__).parents[1] / sparse_name , sparse_FPs_numrcts)
    
    return # sparse_name
    
# if __name__ == '__main__':
    # from pathlib import Path
    # print(Path(__file__).stem) # prints the current filename w/o extensions e.g. smi_to_raw_fp_sparse 
    
    # parse arguments for sparse_from_fp function + path to SMILES dataset + output file name
    # either from terminal, or as parameters into the function 

    