import torch
import torch.tensor as tensor
from torch.utils.data import Dataset
import random
import pickle
import scipy
from scipy import sparse 
import numpy as np
import nmslib
from typing import List, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod

sparse_fp = scipy.sparse.csr_matrix 
dense_fp = np.ndarray # try not to use, more memory intensive

def rcts_prod_fps_from_rxn_smi(rxn_smi: str, fp_type: str, lookup_dict: dict, 
                              mol_fps: sparse_fp) -> Tuple[sparse_fp, sparse_fp]:
    ''' No need to check for KeyError in lookup_dict
    '''
    prod_smi = rxn_smi.split('>')[-1]
    prod_idx = lookup_dict[prod_smi]
    prod_fp = mol_fps[prod_idx]

    if fp_type == 'bit':
        dtype = 'bool' # to add rcts_fp w/o exceeding value of 1
        final_dtype = 'int8' # to cast rcts_fp back into 'int8'
    else: # count
        dtype = 'int16' # infer dtype
        final_dtype = dtype

    rcts_smis = rxn_smi.split('>')[0].split('.')
    for i, rct_smi in enumerate(rcts_smis):
        rct_idx = lookup_dict[rct_smi]
        if i == 0:
            rcts_fp = mol_fps[rct_idx].astype(dtype)
        else:
            rcts_fp = rcts_fp + mol_fps[rct_idx].astype(dtype)
    return rcts_fp.astype(final_dtype), prod_fp

def make_rxn_fp(rcts_fp: sparse_fp, prod_fp: sparse_fp, rxn_type: str='diff') -> sparse_fp:
    ''' 
    Assembles rcts_fp (a sparse array of 1 x fp_size) & prod_fp (another sparse array, usually of the same shape)
    into a reaction fingerprint, rxn_fp, of the fingerprint type requested
    
    rxn_type : str (Default = 'diff')
        currently supports 'diff' & 'sep' fingerprints
    '''
    if rxn_type == 'diff':
        rxn_fp = prod_fp - rcts_fp
    elif rxn_type == 'sep':
        rxn_fp = sparse.hstack([rcts_fp, prod_fp])
    return rxn_fp 

class Augmentor(ABC):
    ''' Abstract Base Class for Augmentors
    '''
    def __init__(self, num_neg: int, lookup_dict: dict, mol_fps: sparse_fp, 
                rxn_type: str, fp_type: str, **kwargs):
        self.num_neg = num_neg
        self.lookup_dict = lookup_dict
        self.mol_fps = mol_fps
        self.rxn_type = rxn_type
        self.fp_type = fp_type

    @abstractmethod
    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        '''
        '''
    @abstractmethod
    def get_idx(self, rxn_smi: str) -> int:
        '''
        '''

class CosineAugmentor:
    '''
    Generates negative reaction fingerprints (on-the-fly) by fetching nearest neighbours 
    (by cosine similarity) of product molecules in a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants
    
    TODO: Currently made for fingerprints - think about how to make this a base class 
    for both fingerprints and graphs. 
    TODO: Allow saving of indices (or the negative fingerprints themselves) on disk
    '''
    def __init__(self, num_neg: int, lookup_dict: dict, 
                mol_fps: sparse_fp, search_index: nmslib.dist.FloatIndex, 
                rxn_type: str='diff', fp_type: str='count', num_threads: int=4):
        self.num_neg = num_neg
        self.lookup_dict = lookup_dict
        self.mol_fps = mol_fps
        self.search_index = search_index # can be None, in which case it'll be initialised by _worker_init_fn_nmslib (see expt.utils)
        self.num_threads = num_threads
        self.rxn_type = rxn_type
        self.fp_type = fp_type
    
    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        ''' 
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        nn_prod_idxs = self.get_idx(prod_fp=prod_fp)
        nn_prod_fps = [ self.mol_fps[idx] for idx in nn_prod_idxs[1: self.num_neg + 1] ] # first index is the original molecule! 
        neg_rxn_fps = []
        for nn_prod_fp in nn_prod_fps:
            neg_rxn_fp = make_rxn_fp(rcts_fp, nn_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp) 
        return neg_rxn_fps
    
    def get_idx(self, prod_fp: sparse_fp=None, rxn_smi: Optional[str]=None) -> int:
        ''' if rxn_smi string is provided, then prod_fp is not needed. otherwise, prod_fp is necessary
        '''
        if rxn_smi:
            rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        nn_prod_idxs_with_dist = self.search_index.knnQueryBatch(prod_fp, k=self.num_neg+1, num_threads=self.num_threads) 
        nn_prod_idxs = nn_prod_idxs_with_dist[0][0] # list of only 1 item, so need one more [0] to access; first item of tuple is np.ndarray of indices 
        return nn_prod_idxs

class RandomAugmentor:
    '''
    Generates negative reaction fingerprints by fetching random product molecules 
    to modify a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants
    
    TODO: Currently made for fingerprints - think about how to make this a base class 
    for both fingerprints and graphs. 
    '''
    def __init__(self, num_neg: int, lookup_dict: dict, 
                mol_fps: sparse_fp, rxn_type: str='diff', fp_type: str='count'):
        self.num_neg = num_neg
        self.lookup_dict = lookup_dict
        self.mol_fps = mol_fps
        self.rxn_type = rxn_type
        self.fp_type = fp_type
        self.orig_idxs = range(self.mol_fps.shape[0])
        # self.rdm_idxs = random.sample(self.orig_idxs, k=len(self.orig_idxs))
        # using this means that a product molecule (if it appears multiple times in different rxns)
        # will consistently map to a particular randomly selected molecule, which might not be a bad thing! 
        # we use the original prod_idx in lookup_dict to access self.rdm_idxs 
        # i.e. w/o this, a product molecule will stochastically map to different molecules throughout training

    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        ''' 
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            rdm_prod_idx = random.choice(self.orig_idxs)
            rdm_prod_fp = self.mol_fps[rdm_prod_idx]
            neg_rxn_fp = make_rxn_fp(rcts_fp, rdm_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp) 
        return neg_rxn_fps 

    def get_idx(self) -> int:
        return random.choice(self.orig_idxs)
        
class BitAugmentor:
    '''
    Generates negative reaction fingerprints by randomly switching the values of bits
    in a given positive reaction fingerprint
    For bit rxn fingerprints, this means randomly replacing values with 1, 0, or -1 
    For count rxn fingerprints, this means randomly adding 1 to the original value at a randomly chosen position

    Future strategies include:
        1) 'Attacks' the most sensitive bits by some analysis function 
            i.e. the bit that the model is currently most sensitive to
        2) Replace with a section of bits from a different molecule 
            (mimics RICAP from computer vision)
        3) Similar to 2, but select the 'donor' molecule using some similarity (or dissimilarity) metric
    NOTE: only compatible with fingerprints
    '''
    def __init__(self, num_neg: int, num_bits: int, strategy: str, 
                lookup_dict: dict, mol_fps: sparse_fp, 
                rxn_type: str='diff', fp_type: str='count'):
        self.num_neg = num_neg
        self.num_bits = num_bits
        self.strategy = strategy
        self.lookup_dict = lookup_dict
        self.mol_fps = mol_fps
        self.rxn_type = rxn_type
        self.fp_type = fp_type

    def get_one_sample_count(self, rxn_smi: str) -> List[sparse_fp]:
        ''' For count fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(range(pos_rxn_fp.shape[-1]), k=self.num_bits) 
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = neg_rxn_fp[0, bit_idx] + 1 
            neg_rxn_fps.append(neg_rxn_fp) 
        return neg_rxn_fps  
    
    def get_one_sample_bit(self, rxn_smi: str) -> List[sparse_fp]:
        ''' For bit fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(range(pos_rxn_fp.shape[-1]), k=self.num_bits) 
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = random.choice([-1, 0, 1])
            neg_rxn_fps.append(neg_rxn_fp) 
        return neg_rxn_fps 

    def get_idx(self, rxn_smi: str) -> None:
        pass

 
