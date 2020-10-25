''' This module contains functions to build and save a nearest-neighbour search index using the sparse
molecular fingerprints of all the unique molecules in a given dataset. Currently, the main
library dependency is nmslib (2.0.6). The constructed index will then be used by CosineAugmentor
during data augmentation and training.

NOTE: nmslib does not support Forking multiprocessing - workarounds are described in data.py
NOTE: nmslib (as of 2.0.6) does not support having multiple vectors of all 0's. We check for those cases,
which happens in RDKit CountFingerprints of very small molecules (like H2O, NH3), and exclude these
molecules from the search index.
NOTE: nmslib doesn't allow me to label the points with something else other than their original index
e.g. label each point in the search index with their SMILES string
'''
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import scipy
from scipy import sparse
from tqdm import tqdm

import nmslib


# def validate_sparse_matrix(
#         sparse_matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
#     ''' only needed for RDKit CountFingerprints (now no longer using!) 
#     '''
#     if sparse_matrix[-1].nonzero()[0].size == 0:
#         print("Truncating last row of sparse_matrix, which is all 0's")
#         return sparse_matrix[:-1]
#     else:
#         return sparse_matrix


def build_nmslib_index(data, method='hnsw', space='cosinesimil_sparse',
                       data_type=nmslib.DataType.SPARSE_VECTOR,
                       num_threads=4, M=30, efC=100):
    print(f'Building index {space} using {method}')
    if space == 'jaccard_sparse':
        data_type = nmslib.DataType.OBJECT_AS_STRING

    spaces_index = nmslib.init(method=method, space=space, data_type=data_type)
    spaces_index.addDataPointBatch(data)

    index_time_params = {
        'M': M,
        'indexThreadQty': num_threads,
        'efConstruction': efC,
        'post': 0}

    start = time.time()
    spaces_index.createIndex(index_time_params)
    end = time.time()
    print(f'Index-time parameters: {index_time_params}')
    print(f'Indexing time = {end-start}')
    return spaces_index


def query_nmslib_index(spaces_index, query_matrix, space='cosinesimil_sparse',
                       num_threads=4, efS=100, K=100):
    if efS:
        query_time_params = {'efSearch': efS}
    else:
        query_time_params = {}
    print('Setting query-time parameters', query_time_params)
    spaces_index.setQueryTimeParams(query_time_params)

    # Querying
    if space == 'jaccard_sparse':
        query_qty = len(query_matrix)
    else:
        query_qty = query_matrix.shape[0]
    start = time.time()
    nbrs = spaces_index.knnQueryBatch(
        query_matrix, k=K, num_threads=num_threads)
    end = time.time()
    print(
        'kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
        (end -
         start,
         float(
             end -
             start) /
            query_qty,
            num_threads *
            float(
             end -
             start) /
            query_qty))
    return nbrs


def build_and_save_index(mol_fps_filename: Union[str,
                                                 bytes,
                                                 os.PathLike] = '50k_count_mol_fps.npz',
                         output_filename: Union[str,
                                                bytes,
                                                os.PathLike] = '50k_cosine_count.bin',
                         root: Optional[Union[str,
                                              bytes,
                                              os.PathLike]] = None):
    '''
    Also see: build_nmslib_index
    '''
    if root is None:
        root = Path(__file__).resolve().parents[2] / 'data' / 'cleaned_data'
    if Path(mol_fps_filename).suffix != '.npz':
        mol_fps_filename = str(mol_fps_filename) + '.npz'
    if Path(output_filename).suffix != '.bin':
        output_filename = str(output_filename) + '.bin'
    if (root / output_filename).exists():
        print(f'At: {root / output_filename}')
        print('The search index file already exists!') 
        return

    mol_fps = sparse.load_npz(root / mol_fps_filename)
    mol_fps_validated = validate_sparse_matrix(mol_fps)
    index = build_nmslib_index(
        method='hnsw',
        space='cosinesimil_sparse',
        data=mol_fps_validated,
        M=30,
        efC=100)
    index.saveIndex(str(root / output_filename), save_data=True)
    print('Successfully built and saved index!')


if __name__ == '__main__':
    build_and_save_index()  # defaults to countfps
