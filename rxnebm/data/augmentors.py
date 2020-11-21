import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy
from scipy import sparse

import nmslib
from rxnebm.data.preprocess import smi_to_fp

sparse_fp = scipy.sparse.csr_matrix
dense_fp = np.ndarray  # try not to use, more memory intensive

""" TODO: split precomputation into one .npz file of just random negatives, one .npz file of just cosine negatives,
and one .npz file of just bit negatives, so that if we want to modify certain augmentation hyperparams midway through training,
e.g. start with just random & cosine --> include random, cosine & bit --> increase num_neg of bit etc.,
separately stored .npz files will make it much easier.
"""


def rcts_prod_fps_from_rxn_smi(
    rxn_smi: str, fp_type: str, smi_to_fp_dict: dict, mol_fps: sparse_fp
) -> Tuple[sparse_fp, sparse_fp]:
    prod_smi = rxn_smi.split(">")[-1]
    prod_idx = smi_to_fp_dict[prod_smi]
    prod_fp = mol_fps[prod_idx]

    if fp_type == "bit":
        dtype = "bool"  # to add rcts_fp w/o exceeding value of 1
        final_dtype = "int32"  # to cast rcts_fp back into 'int8'
    else:  # count
        dtype = "int32"  # or maybe to infer dtype, but that adds computation overhead
        final_dtype = dtype

    rcts_smis = rxn_smi.split(">")[0].split(".")
    for i, rct_smi in enumerate(rcts_smis):
        rct_idx = smi_to_fp_dict[rct_smi]
        if i == 0:
            rcts_fp = mol_fps[rct_idx].astype(dtype)
        else:
            rcts_fp = rcts_fp + mol_fps[rct_idx].astype(dtype)
    return rcts_fp.astype(final_dtype), prod_fp


def make_rxn_fp(
    rcts_fp: sparse_fp, prod_fp: sparse_fp, rxn_type: str = "diff"
) -> sparse_fp:
    """
    Assembles rcts_fp (a sparse array of 1 x fp_size) & prod_fp (another sparse array, usually of the same shape)
    into a reaction fingerprint, rxn_fp, of the fingerprint type requested

    rxn_type : str (Default = 'diff')
        currently supports 'diff' & 'sep' fingerprints
    """
    if rxn_type == "diff":
        rxn_fp = prod_fp - rcts_fp
    elif rxn_type == "sep":
        rxn_fp = sparse.hstack([rcts_fp, prod_fp])
    return rxn_fp


class Augmentor(ABC):
    """Abstract Base Class for Augmentors"""

    def __init__(
        self,
        num_neg: int, 
        smi_to_fp_dict: dict,
        mol_fps: sparse_fp,
        rxn_type: str,
        fp_type: str,
        **kwargs
    ):
        self.num_neg = num_neg 
        self.smi_to_fp_dict = smi_to_fp_dict
        self.mol_fps = mol_fps
        self.rxn_type = rxn_type
        self.fp_type = fp_type

    @abstractmethod
    def get_one_sample(self, rxn_smi: str) -> List[str]:
        """ Generate self.num_neg * negative_rxn_smi from a given rxn_smi
        """
        raise NotImplementedError

    def get_one_sample_fp(self, rxn_smi: str) -> List[sparse_fp]:
        """ Generate self.num_neg * negative_rxn_fp from a given rxn_smi
        """
        pass

    @abstractmethod
    def get_idx(self, rxn_smi: Optional[str] = None) -> int:
        """"""

class Mutate(Augmentor):
    """NOTE: READS MUTATED SMIS FROM PRECOMPUTED DICTIONARY ON DISK

    Adapted from CReM: https://github.com/DrrDom/crem
    to mutate product molecules into structurally-similar molecules

    Ideas (maybe TODO)
        1) divide num_neg into K splits, where each split comes from mutate_mols() set up with specific sets of hyperparams
        (see crem_example notebook from his github repo)
        e.g. one split can be dedicated to min_inc & max_inc both -ve (shrink orig mol), another split with both +ve (expand orig mol)
        then use random.sample(mutated_mols, num_neg // K)
        this way, we get a wider variety of mutated mols in each minibatch of negative examples
    """

    def __init__(
        self,
        num_neg: int,
        mut_smis: dict, 
        smi_to_fp_dict: Optional[dict] = None, 
        mol_fps: Optional[sparse_fp] = None,
        rxn_type: Optional[str] = "diff",
        fp_type: Optional[str] = "count",
        radius: Optional[int] = 3,
        fp_size: Optional[int] = 4096,
        dtype: Optional[str] = "int32",
    ):
        super(Mutate, self).__init__(num_neg, smi_to_fp_dict, mol_fps, rxn_type, fp_type)
        self.mut_smis = mut_smis
        self.radius = radius
        self.fp_size = fp_size
        self.dtype = dtype

        if smi_to_fp_dict: # means using fingerprints
            self.set_mol_smi_to_fp_func()

    def set_mol_smi_to_fp_func(self):
        ''' Called by self.get_one_sample_fp()
        ''' 
        if self.fp_type == "count":
            self.mol_smi_to_fp = smi_to_fp.mol_smi_to_count_fp
        elif self.fp_type == "bit":
            self.mol_smi_to_fp = smi_to_fp.mol_smi_to_bit_fp 

    # def set_get_one_sample(self):
    #     if self.get_fp:
    #         self.get_one_sample = self.get_one_sample_fp
    #     else:
    #         self.get_one_sample = self.get_one_sample_smi

    def get_one_sample(self, rxn_smi: str) -> List[str]:
        prod_smi = rxn_smi.split(">>")[-1]
        rcts_smi = rxn_smi.split('>>')[0]
        mut_prod_smis = self.mut_smis[prod_smi]
        if len(mut_prod_smis) > self.num_neg:
            mut_prod_smis = random.sample(mut_prod_smis, self.num_neg)
 
        neg_rxn_smis = []
        for mut_prod_smi in mut_prod_smis:
            neg_rxn_smi = rcts_smi + '>>' + mut_prod_smi
            neg_rxn_smis.append(neg_rxn_smi)
 
        if len(neg_rxn_smis) < self.num_neg: 
            for i in range(self.num_neg - len(neg_rxn_smis)):
                neg_rxn_smis.append('') # pad with '' whenever insufficient negs generated by CReM

        return neg_rxn_smis

    def get_one_sample_fp(self, rxn_smi: str) -> List[sparse_fp]:
        """ Generates a bunch of negative reaction fingerprints from a given reaction SMILES string
        Called directly by dataset.AugmentedDataFingerprints.get_one_minibatch() 

        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp, smi_to_fp.mol_smi_to_count_fp()
        """
        prod_smi = rxn_smi.split(">>")[-1]
        mut_prod_smis = self.mut_smis[prod_smi]
        if len(mut_prod_smis) > self.num_neg:
            mut_prod_smis = random.sample(mut_prod_smis, self.num_neg)

        mut_prod_fps = []
        for mut_prod_smi in mut_prod_smis:
            prod_fp = self.mol_smi_to_fp(
                mut_prod_smi, self.radius, self.fp_size, self.dtype
            )
            mut_prod_fps.append(prod_fp)

        rcts_fp, _ = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
        )

        neg_rxn_fps = []
        for mut_prod_fp in mut_prod_fps:
            neg_rxn_fp = make_rxn_fp(rcts_fp, mut_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp)
 
        if len(neg_rxn_fps) < self.num_neg: 
            dummy_fp = np.zeros((1, self.fp_size), dtype=self.dtype) # pad with zero-vectors whenever insufficient negs generated from CReM
            neg_rxn_fps.extend([dummy_fp] * (self.num_neg - len(neg_rxn_fps)))

        return neg_rxn_fps

    def get_idx(self, mut_prod_smis: List[str]) -> None:
        """Not applicable to CReM
        Because need to expand smi_to_fp_dict to include the newly generated negative molecules, and
        there are too many to keep track!
        """
        raise NotImplementedError('Mutate Augmentor does not support get_idx() method!') 

class Cosine(Augmentor):
    """
    Generates negative reaction fingerprints (on-the-fly) by fetching nearest neighbours
    (by cosine similarity) of product molecules in a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants

    TODO: Currently made for fingerprints - think about how to make this a base class
    for both fingerprints and graphs 
    """

    def __init__(
        self,
        num_neg: int,
        search_index: nmslib.dist.FloatIndex,
        smi_to_fp_dict: dict,
        fp_to_smi_dict: dict, 
        mol_fps: sparse_fp,
        rxn_type: str = "diff",
        fp_type: str = "count",
        num_threads: int = 4,
    ):
        super(Cosine, self).__init__(num_neg, smi_to_fp_dict, mol_fps, rxn_type, fp_type)
        self.fp_to_smi_dict = fp_to_smi_dict
        
        # can be None, in which case it'll be initialised by _worker_init_fn_nmslib (see expt.utils)
        self.search_index = search_index
        self.num_threads = num_threads
    
    def get_one_sample(self, rxn_smi: str) -> List[str]:
        prod_smi = rxn_smi.split(">>")[-1]
        rcts_smi = rxn_smi.split('>>')[0]
 
        nn_prod_idxs = self.get_idx(prod_fp=prod_fp, rxn_smi=rxn_smi)
        nn_prod_smis = [self.fp_to_smi_dict[idx] for idx in nn_prod_idxs[: self.num_neg]]

        neg_rxn_smis = []
        for nn_prod_smi in nn_prod_smis:
            neg_rxn_smi = rcts_smi + '>>' + nn_prod_smi
            neg_rxn_smis.append(neg_rxn_smi)

        return neg_rxn_smis

    def get_one_sample_fp(self, rxn_smi: str) -> List[sparse_fp]:
        """
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        """
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
        )
        
        nn_prod_idxs = self.get_idx(prod_fp=prod_fp, rxn_smi=rxn_smi)
        nn_prod_fps = [self.mol_fps[idx] for idx in nn_prod_idxs[: self.num_neg]]

        neg_rxn_fps = []
        for nn_prod_fp in nn_prod_fps:
            neg_rxn_fp = make_rxn_fp(rcts_fp, nn_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_idx(self, prod_fp: sparse_fp, rxn_smi: str) -> List[int]:
        """ if prod_fp is None, will create it from rxn_smi 
        """
        if prod_fp is None:
            rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
                rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
            )

        nn_prod_idxs_with_dist = self.search_index.knnQueryBatch(
            prod_fp, k=self.num_neg + 3, num_threads=self.num_threads
        )

        all_rct_smis = set(rxn_smi.split('>>')[0].split('.'))
        nn_prod_idxs = nn_prod_idxs_with_dist[0][0]
        nn_prod_idxs_clean = []
        for prod_idx in nn_prod_idxs[1:]: # 0th element = orig prod; we don't want it as negative
            new_prod_smi = self.fp_to_smi_dict[prod_idx]
            if new_prod_smi in all_rct_smis: # prod == one of the rcts, do not use this prod!
                continue
            else:
                nn_prod_idxs_clean.append(prod_idx)

        return nn_prod_idxs_clean 

    def get_nn_prod_smis(self, prod_fp: sparse_fp, rxn_smi: str) -> List[str]:
        if prod_fp is None:
            rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
                rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
            )
        
        nn_prod_idxs = self.get_idx(prod_fp, rxn_smi)

        nn_prod_smis = []
        for prod_idx in nn_prod_idxs:  
            nn_prod_smi = self.fp_to_smi_dict[prod_idx]
            nn_prod_smis.append(nn_prod_smi)

        return nn_prod_smis 


class Random(Augmentor):
    """
    Generates negative reaction fingerprints by fetching random product molecules
    to modify a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants

    TODO: Currently made for fingerprints - think about how to make this a base class
    for both fingerprints and graphs.
    """

    def __init__(
        self,
        num_neg: int,
        smi_to_fp_dict: dict,
        fp_to_smi_dict: dict, 
        mol_fps: sparse_fp,
        rxn_type: str = "diff",
        fp_type: str = "count",
    ):
        super(Random, self).__init__(num_neg, smi_to_fp_dict, mol_fps, rxn_type, fp_type)
        self.fp_to_smi_dict = fp_to_smi_dict
        self.orig_idxs = range(self.mol_fps.shape[0])
        self.rdm_idxs = random.sample(self.orig_idxs, k=len(self.orig_idxs))
        self.rdm_counter = -1 

    def get_one_sample(self, rxn_smi: str) -> List[str]:
        prod_smi = rxn_smi.split(">>")[-1]
        rcts_smi = rxn_smi.split('>>')[0]
 
        neg_rxn_smis = []
        for i in range(self.num_neg):
            rdm_prod_idx = self.get_idx() 
            rdm_prod_smi = self.fp_to_smi_dict[rdm_prod_idx] 
            neg_rxn_smi = rcts_smi + '>>' + rdm_prod_smi
            neg_rxn_smis.append(neg_rxn_smi)

        return neg_rxn_smis

    def get_one_sample_fp(self, rxn_smi: str) -> List[sparse_fp]:
        """
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        """
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
        )

        neg_rxn_fps = []
        for i in range(self.num_neg):
            rdm_prod_idx = self.get_idx() 
            rdm_prod_fp = self.mol_fps[rdm_prod_idx]
            neg_rxn_fp = make_rxn_fp(rcts_fp, rdm_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_idx(self) -> int:
        self.rdm_counter += 1
        self.rdm_counter %= self.mol_fps.shape[0]
        return self.rdm_idxs[self.rdm_counter]  

class Bit(Augmentor):
    """
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
    """

    def __init__(
        self,
        num_neg: int,
        num_bits: int,
        increment_bits: int,
        strategy: str,
        smi_to_fp_dict: dict,
        mol_fps: sparse_fp,
        rxn_type: str = "diff",
        fp_type: str = "count",
    ):
        super(Bit, self).__init__(num_neg, smi_to_fp_dict, mol_fps, rxn_type, fp_type)
        self.num_bits = num_bits
        self.increment_bits = increment_bits or 1  # default is 1
        self.strategy = strategy

        self.set__get_one_sample_func()

    def get_one_sample_smi(self, rxn_smi: str) -> None:
        raise NotImplementedError('Bit Augmentor not able to generate neg_rxn_smi!!')
     
    def set__get_one_sample_func(self):
        ''' self.get_one_sample & self.get_one_sample_fp are identical 
        '''
        if self.fp_type == "count":
            self.get_one_sample = self.get_one_sample_count
            self.get_one_sample_fp = self.get_one_sample_count
        elif self.fp_type == "bit":
            self.get_one_sample = self.get_one_sample_bit
            self.get_one_sample_fp = self.get_one_sample_count

    def get_one_sample(self, rxn_smi: str) -> List[sparse_fp]:
        """indirectly set by self.set__get_one_sample_fp_func()"""
        raise RuntimeError('This should never be executed!')
        pass 

    def get_one_sample_count(self, rxn_smi: str) -> List[sparse_fp]:
        """For count fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        """
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
        )
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(range(pos_rxn_fp.shape[-1]), k=self.num_bits)
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = neg_rxn_fp[0, bit_idx] + self.increment_bits
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_one_sample_bit(self, rxn_smi: str) -> List[sparse_fp]:
        """For bit fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        """
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
        )
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(range(pos_rxn_fp.shape[-1]), k=self.num_bits)
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = random.choice([-1, 0, 1])
            neg_rxn_fps.append(neg_rxn_fp)
        return neg_rxn_fps

    def get_idx(self) -> None:
        pass
