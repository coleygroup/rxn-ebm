import torch
import torch.tensor as tensor
from torch.utils.data import Dataset
import random
import pickle
import scipy
from scipy import sparse
import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod
import sqlite3
 
import rdkit
from rdkit import Chem
import nmslib

sparse_fp = scipy.sparse.csr_matrix
dense_fp = np.ndarray  # try not to use, more memory intensive


def rcts_prod_fps_from_rxn_smi(rxn_smi: str,
                               fp_type: str,
                               lookup_dict: dict,
                               mol_fps: sparse_fp) -> Tuple[sparse_fp,
                                                            sparse_fp]:
    ''' No need to check for KeyError in lookup_dict
    '''
    prod_smi = rxn_smi.split('>')[-1]
    prod_idx = lookup_dict[prod_smi]
    prod_fp = mol_fps[prod_idx]

    if fp_type == 'bit':
        dtype = 'bool'  # to add rcts_fp w/o exceeding value of 1
        final_dtype = 'int8'  # to cast rcts_fp back into 'int8'
    else:  # count
        dtype = 'int16'  # infer dtype
        final_dtype = dtype

    rcts_smis = rxn_smi.split('>')[0].split('.')
    for i, rct_smi in enumerate(rcts_smis):
        rct_idx = lookup_dict[rct_smi]
        if i == 0:
            rcts_fp = mol_fps[rct_idx].astype(dtype)
        else:
            rcts_fp = rcts_fp + mol_fps[rct_idx].astype(dtype)
    return rcts_fp.astype(final_dtype), prod_fp


def make_rxn_fp(
        rcts_fp: sparse_fp,
        prod_fp: sparse_fp,
        rxn_type: str = 'diff') -> sparse_fp:
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
        raise NotImplementedError

    @abstractmethod
    def get_idx(self, rxn_smi: Optional[str] = None) -> int:
        '''
        '''
        raise NotImplementedError


class MutateAugmentor(Augmentor):
    ''' NOTE: READS MUTATED SMIS FROM PRECOMPUTED DICTIONARY ON DISK

    Adapted from CReM: https://github.com/DrrDom/crem 
    to mutate product molecules into structurally-similar molecules
    TODO: make frag_db from USPTO_50k? for now, use the one CReM already comes with

    Ideas:
        1) divide num_neg into K splits, where each split comes from mutate_mols() set up with specific sets of hyperparams
        (see crem_example notebook from his github repo)
        e.g. one split can be dedicated to min_inc & max_inc both -ve (shrink orig mol), another split with both +ve (expand orig mol)
        then use random.sample(mutated_mols, num_neg // K)
        this way, we ensure a variety of mutated mols in each minibatch of negative examples
    '''

    def __init__(self,
                 num_neg: int,
                 lookup_dict: dict,
                 mol_fps: sparse_fp,
                 mut_smis_filename: Union[str, bytes, os.PathLike],
                 root: Optional[Union[str, bytes, os.PathLike]] = None,
                 rxn_type: str = 'diff',
                 fp_type: str = 'count'):
        super(
            MutateAugmentor,
            self).__init__(
            num_neg,
            lookup_dict,
            mol_fps,
            rxn_type,
            fp_type)

        if root is None:
            root = Path(__file__).parents[1] / 'data' / 'cleaned_data'
        if Path(mut_smis_filename).suffix != '.pickle':
            mut_smis_filename = str(mut_smis_filename) + '.pickle'
        with open(root / mut_smis_filename, 'rb') as handle:
            self.mut_smis = pickle.load(handle)

    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        '''
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        prod_smi = rxn_smi.split('>>')[-1]
        mut_prod_smis = mut_smis[prod_smi] # load mut_prods from disk 

        mut_prod_fps = []
        for prod_idx in self.get_idx(mut_prod_smis):
            prod_fp = self.mol_fps[prod_idx]
            mut_prod_fps.append(prod_fp)

        rcts_fp, _ = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)

        neg_rxn_fps = []
        for mut_prod_fp in mut_prod_fps:
            neg_rxn_fp = make_rxn_fp(rcts_fp, mut_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_idx(self, mut_prod_smis: List[str]) -> List[int]:
        prod_idxs = []
        if len(mut_prod_smis) < self.num_neg: # fill up to self.num_neg using random sampling 
            prod_idxs = random.sample(range(len(self.lookup_dict)), self.num_neg - len(mut_prod_smis))            
        else:
            for prod_smi in random.sample(mut_prod_smis, self.num_neg):
                prod_idxs.append(self.lookup_dict[prod_smi])
        return prod_idxs


class CosineAugmentor(Augmentor):
    '''
    Generates negative reaction fingerprints (on-the-fly) by fetching nearest neighbours
    (by cosine similarity) of product molecules in a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants

    TODO: Currently made for fingerprints - think about how to make this a base class
    for both fingerprints and graphs.
    TODO: Allow saving of indices (or the negative fingerprints themselves) on disk
    '''

    def __init__(
            self,
            num_neg: int,
            lookup_dict: dict,
            mol_fps: sparse_fp,
            search_index: nmslib.dist.FloatIndex,
            rxn_type: str = 'diff',
            fp_type: str = 'count',
            num_threads: int = 4):
        super(
            CosineAugmentor,
            self).__init__(
            num_neg,
            lookup_dict,
            mol_fps,
            rxn_type,
            fp_type)
        # can be None, in which case it'll be initialised by
        # _worker_init_fn_nmslib (see expt.utils)
        self.search_index = search_index
        self.num_threads = num_threads

    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        '''
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        nn_prod_idxs = self.get_idx(prod_fp=prod_fp)
        # first index is the original molecule!
        nn_prod_fps = [self.mol_fps[idx]
                       for idx in nn_prod_idxs[1: self.num_neg + 1]]
        neg_rxn_fps = []
        for nn_prod_fp in nn_prod_fps:
            neg_rxn_fp = make_rxn_fp(rcts_fp, nn_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_idx(self, prod_fp: sparse_fp = None,
                rxn_smi: Optional[str] = None) -> int:
        ''' if rxn_smi string is provided, then prod_fp is not needed.
        otherwise, prod_fp is necessary
        '''
        if rxn_smi:
            rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
                rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        nn_prod_idxs_with_dist = self.search_index.knnQueryBatch(
            prod_fp, k=self.num_neg + 1, num_threads=self.num_threads)

        # list of only 1 item, so need one more [0] to access; first item of
        # tuple is np.ndarray of indices
        nn_prod_idxs = nn_prod_idxs_with_dist[0][0]
        return nn_prod_idxs


class RandomAugmentor(Augmentor):
    '''
    Generates negative reaction fingerprints by fetching random product molecules
    to modify a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants

    TODO: Currently made for fingerprints - think about how to make this a base class
    for both fingerprints and graphs.
    '''

    def __init__(
            self,
            num_neg: int,
            lookup_dict: dict,
            mol_fps: sparse_fp,
            rxn_type: str = 'diff',
            fp_type: str = 'count'):
        super(
            RandomAugmentor,
            self).__init__(
            num_neg,
            lookup_dict,
            mol_fps,
            rxn_type,
            fp_type)
        self.orig_idxs = range(self.mol_fps.shape[0])
        self.rdm_idxs = random.sample(self.orig_idxs, k=len(self.orig_idxs))
        self.rdm_counter = -1
        # using this means that a product molecule (if it appears multiple times in different rxns)
        # will consistently map to a particular randomly selected molecule, which might not be a bad thing!
        # we use the original prod_idx in lookup_dict to access self.rdm_idxs
        # i.e. w/o this, a product molecule will stochastically map to
        # different molecules throughout training

    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        '''
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            rdm_prod_idx = self.get_idx()  # random.choice(self.orig_idxs)
            rdm_prod_fp = self.mol_fps[rdm_prod_idx]
            neg_rxn_fp = make_rxn_fp(rcts_fp, rdm_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_idx(self) -> int:
        self.rdm_counter += 1
        return self.rdm_idxs[self.rdm_counter]
        # return random.choice(self.orig_idxs)


class BitAugmentor(Augmentor):
    '''
    Generates negative reaction fingerprints by randomly switching the values of bits
    in a given positive reaction fingerprint
    For bit rxn fingerprints, this means randomly replacing values with 1, 0, or -1
    For count rxn fingerprints, this means randomly adding 1 to the original value at a randomly chosen position

    Future strategies include:
        1) 'Attacks' the most sensitive bits by some analysis function??
            i.e. the bit that the model is currently most sensitive to
        2) Replace with a section of bits from a different molecule
            (mimics RICAP from computer vision)
        3) Similar to 2, but select the 'donor' molecule using some similarity (or dissimilarity) metric
    NOTE: only compatible with fingerprints
    '''

    def __init__(
            self,
            num_neg: int,
            num_bits: int,
            increment_bits: int,
            strategy: str,
            lookup_dict: dict,
            mol_fps: sparse_fp,
            rxn_type: str = 'diff',
            fp_type: str = 'count'):
        super(
            BitAugmentor,
            self).__init__(
            num_neg,
            lookup_dict,
            mol_fps,
            rxn_type,
            fp_type)
        self.num_bits = num_bits
        self.increment_bits = increment_bits or 1  # default is 1
        self.strategy = strategy

    def get_one_sample_count(self, rxn_smi: str) -> List[sparse_fp]:
        ''' For count fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(
                range(pos_rxn_fp.shape[-1]), k=self.num_bits)
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = neg_rxn_fp[0,
                                                    bit_idx] + self.increment_bits
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_one_sample_bit(self, rxn_smi: str) -> List[sparse_fp]:
        ''' For bit fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(
                range(pos_rxn_fp.shape[-1]), k=self.num_bits)
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = random.choice([-1, 0, 1])
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        if self.fp_type == 'count':
            return self.get_one_sample_count(rxn_smi)
        else:
            return self.get_one_sample_bit(rxn_smi)

    def get_idx(self) -> None:
        pass
