import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nmslib
import numpy as np
import rdkit
import scipy
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import (GetMorganGenerator,
                                               GetRDKitFPGenerator)
from scipy import sparse
from tqdm import tqdm

sparse_fp = scipy.sparse.csr_matrix

def mol_smi_to_count_fp(
    mol_smi: str, radius: int = 3, fp_size: int = 4096, dtype: str = "int32"
) -> scipy.sparse.csr_matrix:
    fp_gen = GetMorganGenerator(
        radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size
    )
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)


def rcts_prod_fps_from_rxn_smi_dist(
    rxn_smi: str, radius: int, fp_size: int
) -> Tuple[sparse_fp, sparse_fp]:
    ''' 
    The idea is not to pickle the memory-heavy count_mol_fps sparse matrix across processes
    We trade that memory for compute, since now we have to make each count mol fp on the fly
    But hopefully when multiprocessing w/ enough processes (say 32 or 64), 
    the parallelization should overall speed up the whole process

    Indeed, this way is much, much faster than pre-computing molecular count_fps & assembling the rxn_fp
    (even w/ multi-processing)
    '''
    dtype = "int32" 
    final_dtype = dtype

    prod_smi = rxn_smi.split(">>")[-1]
    prod_fp = mol_smi_to_count_fp(prod_smi, radius, fp_size, dtype)

    rcts_smis = rxn_smi.split(">>")[0].split(".")
    for i, rct_smi in enumerate(rcts_smis):
        if i == 0:
            rcts_fp = mol_smi_to_count_fp(rct_smi, radius, fp_size, dtype)
        else:
            rcts_fp = rcts_fp + mol_smi_to_count_fp(rct_smi, radius, fp_size, dtype)
    return rcts_fp.astype(final_dtype), prod_fp

def make_rxn_fp(
    rcts_fp: sparse_fp, prod_fp: sparse_fp, rxn_type: str = "diff"
) -> sparse_fp:
    """
    Assembles rcts_fp (a sparse array of 1 x fp_size) & prod_fp (another sparse array, usually of the same shape)
    into a reaction fingerprint, rxn_fp, of the fingerprint type requested

    rxn_type : str (Default = 'diff') ['diff', 'sep', 'hybrid', 'hybrid_all']
        'diff': 
            subtract rcts_fp (sum of all rct_fp for that rxn) from prod_fp
        'sep':
            simply concatenate rcts_fp with prod_fp
        'hybrid':
            concatenate prod_fp with 'diff' fp 
        'hybrid_all':
            concatenate rcts_fp, prof_fp & 'diff' fp 
    """
    if rxn_type == "diff":
        rxn_fp = prod_fp - rcts_fp
    elif rxn_type == "sep":
        rxn_fp = sparse.hstack([rcts_fp, prod_fp])
    elif rxn_type == 'hybrid':
        diff_fp = prod_fp - rcts_fp
        rxn_fp = sparse.hstack([prod_fp, diff_fp])
    elif rxn_type == 'hybrid_all':
        diff_fp = prod_fp - rcts_fp
        rxn_fp = sparse.hstack([rcts_fp, prod_fp, diff_fp])
    return rxn_fp
