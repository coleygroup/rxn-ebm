import pysparnn.cluster_index as ci
from scipy import sparse

import sys
import os

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from itertools import chain
import random
from tqdm import tqdm
import pickle
import time

from sklearn.neighbors import NearestNeighbors

import nmslib
 
def build_spaces_index(data, method='hnsw', space='cosinesimil_sparse', 
                       data_type=nmslib.DataType.SPARSE_VECTOR,
                       num_threads=4, 
                       M=30, efC=100):
    if space=='jaccard_sparse':
        data_type = nmslib.DataType.OBJECT_AS_STRING
    spaces_index = nmslib.init(method=method, space=space, data_type=data_type) 
    spaces_index.addDataPointBatch(data)

    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}

    start = time.time()
    spaces_index.createIndex(index_time_params) 
    end = time.time() 
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end-start))
    
    return spaces_index


def query_spaces_index(spaces_index, space='cosinesimil_sparse',
                       query_matrix=mol_FPs_sparse_50k[0:100], 
                       num_threads=4, 
                       efS=100, K=100):
    if efS:
        query_time_params = {'efSearch': efS}
    else:
        query_time_params = {}
    print('Setting query-time parameters', query_time_params)
    spaces_index.setQueryTimeParams(query_time_params)

    # Querying
    if space=='jaccard_sparse':
        query_qty = len(query_matrix)
    else:
        query_qty = query_matrix.shape[0]
    start = time.time() 
    nbrs = spaces_index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
    end = time.time() 
    print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
          (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty))

    return nbrs

def main():
    root = Path(__file__).parents[1]
    sparse_filename = '50k_all_mols_sparse_FPs.npz'
    output_filename = 'spaces_cosinesimil_index.bin'

    mol_FPs_sparse_50k = sparse.load_npz(root / 'cleaned_data' / sparse_filename)

    cosine_index = build_spaces_index(method='hnsw', space='cosinesimil_sparse', data=mol_FPs_sparse_50k, 
                                               M=30, efC=100)
    cosine_index.saveIndex(root / 'cleaned_data' / output_filename, save_data=True)
    print('successfully built and saved index!')