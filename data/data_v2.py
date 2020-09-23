import torch
from torch.utils.data import Dataset
import random
import pickle
from scipy import sparse 
import numpy as np
import nmslib
from typing import List, Optional

def mol_fps_to_rxn_fp(mol_fps: List[np.ndarray], fp_type: Optional[str] = 'diff'
                     ) -> np.ndarray or Tuple[np.ndarray, np.ndarray]:
    '''
    Parameters
    ----------
    mol_fps : List[np.ndarray]
        list of np.ndarray fingerprints of all reactant molecules and the product molecule in a reaction
        (current assumption: only 1 product molecule per reaction)

    fp_type : Optional[str] (Default = 'diff')
        'diff' (difference):
        Creates difference reaction fingerprint 
        rxn_fp = prod_fp - sum(rct_fps)
        Reference: Schneider et al, J. Chem. Inf. Model. 2015, 55, 1, 39–53
        
        'sep' (separate):
        Creates separate sum(rcts_fp) and prod_fp, then concatenates them 
        Reference: Gao et al, ACS Cent. Sci. 2018, 4, 11, 1465–1476
    
    Returns
    -------
    rxn_fp : np.ndarray
        The processed reaction fingerprint, ready to be augmented (e.g. bit corruption) 
        or fed into the model 

        NOTE: the length of a 'sep' fingerprint will be the sum of 
        the length of a reactant fingerprint and the length of a product fingerprint
    '''    
    # access product fingerprint
    if fp_type == 'diff':
        diff_fp = mol_fps[-1]
    elif fp_type == 'sep':
        prod_fp = mol_fps[-1]
                                  
    # access reactant fingerprints
    for i in range(len(mol_fps[: -1])):
        if fp_type == 'diff': # subtract each reactant fingerprint from the product fingerprint
            diff_fp = diff_fp - mol_fps[i]
        elif fp_type == 'sep': # sum the reactant fingerprints
            rcts_fp = rcts_fp + mol_fps[i]
    
    if fp_type == 'diff':
        return diff_fp
    elif fp_type == 'sep':
        return np.concatenate([rcts_fp, prod_fp], axis=1) # can we avoid np.concatenate? 

class AugmentedFingerprints(Dataset):
    '''
    AugmentedFingerprints is a PyTorch Dataset class to prepare training samples of K reaction fingerprints: 
    [positive_rxn, negative_rxn_1, ..., negative_rxn_K-1] where K-1 = number of negative examples per reaction

    Parameters
    ----------
    trainargs : dict
        dictionary of hyperparameters for the current experiment

    dataset : str 
        whether we are using train, valid, or test dataset 
        used to append to all base_paths to get the full filepaths 
    
    base_path_rxnsmi : str
        base path to .pickle file containing reaction SMILES strings
        should have the form: 'path/to/file/prefix_of_rxnsmi_pickle' 
        e.g. 'USPTO_50k_data/clean_rxn_50k_nomap_noreagent' 
        internally, we append '_{dataset}.pickle' to this, e.g. '_train.pickle'

    augs : List[str]
        list of augmentations to apply 
        currently supported options: 'random', 'cosine', 'bit' 
    
    aug_params : dict # List[Tuple[int, int]]
        dictionary of parameters for the augmentations specified in augs, must be in the same order as augs 
        for random sampling: 
            rdm_neg_prod, rdm_neg_rct : #  Optional[Tuple[int, int]]
                rdm_neg_prod : Number of product molecules to randomly sample and replace
                rdm_neg_rct : Number of reactant molecules to randomly sample and replace

        for cosine sampling:
            cos_neg_prod, cos_neg_rct : #  Optional[Tuple[int, int]]
                cos_neg_prod : Number of nearest neighbour product molecules to sample and replace
                cos_neg_rct : Number ofnearest neighbour reactant molecules to sample and replace            

        for bit corruption:
            bit_neg_rxn, bit_neg_bits : # Optional[Tuple[int, int]]]
                bit_neg_rxn : Number of negative reaction fingerprints to generate
                bit_neg_bits : Number of bits to randomly increase or decrease by 1 in each reaction fingerprint 

    Additional parameters
    --------------------
    For random sampling: 
        path_rct_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of reactant molecules 
            should have the form: 'path/to/file/rctfps_filename.npz' 
        path_prod_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of product molecules 
            should have the form: 'path/to/file/prodfps_filename.npz'     
            
    For cosine sampling:
        path_rct_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of reactant molecules 
            should have the form: 'path/to/file/rctfps_filename.npz' 
        path_prod_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of product molecules 
            should have the form: 'path/to/file/prodfps_filename.npz'     
        base_path_nnindex: Optional[str] (Default = None)
            path to .bin file containing the nearest neighbour search index on all the molecules across train, valid and test
            should have the form: 'path/to/file/prefix_of_search_index' 
            e.g. 'USPTO_50k_data/clean_rxn_50k_sparse_FPs'
            internally, we append f'_{dataset}.npz' to this, e.g. '_train.npz'

    For bit corruption:
        base_path_rxnfp: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of reactions
            should have the form 'path/to/file/prefix_of_rxnfp' 
            e.g. 'USPTO_50k_data/clean_rxn_50k_sparse_rxnFPs'
            internally, we append f'_{dataset}.npz' to this, e.g. '_train.npz'
    
    Also see: mol_fps_to_rxn_fp
    '''
    def __init__(self, trainargs: dict, dataset: str, base_path_rxnsmi: str,
                augs = List[str], aug_params = dict, # List[Tuple[int, int]],
                path_rct_fps: Optional[str] = None, path_prod_fps: Optional[str] = None, 
                base_path_nnindex: Optional[str] = None, base_path_rxnfp: Optional[str] = None): 
        self.trainargs = trainargs
        self.fp_type = self.trainargs['fp_type'] # needed for mol_fps_to_rxn_fp
        self.dataset = dataset

        for aug in augs:
            if aug == 'bit':
                full_path_rxnfp = base_path_rxnfp + f'_{dataset}.npz'
                self.rxn_fps = sparse.load_npz(full_path_rxnfp)
                self.rxn_fps_length = self.rxn_fps.shape[0]
                self.bit_neg_rxn = aug_params['bit_neg_rxn']
                self.bit_neg_bits = aug_params['bit_neg_bits']
            
            elif aug == 'cosine':
                self.rct_fps = sparse.load_npz(path_rct_fps)
                self.prod_fps = sparse.load_npz(path_prod_fps)
                self.mol_fps = sparse.load_npz()
                full_path_nnindex = base_path_nnindex + f'_{dataset}.npz'
                
         
            self.fp_raw_num_rcts = sparse.load_npz(base_path + '_' + dataset + '.npz')
            self.fp_raw_num_rcts_length = self.fp_raw_num_rcts.shape[0]           
            self.max_num_rcts = self.fp_raw_num_rcts[0].toarray()[:, :-1].shape[1] // self.rctfp_size - 1
            # to speed up faster_vstack & np.reshape

#         self.show_neg = show_neg

    def _init_searchindex(self):
        self.sparseFP_vocab = sparse.load_npz(self.trainargs['sparseFP_vocab_path'])
        self.clusterindex = None  # if multiprocessing, this will be initialised from worker_init_fn
        if self.trainargs['num_workers'] == 0:
            self._load_nmslib()

    def _load_nmslib(self): 
        self.clusterindex = nmslib.init(method='hnsw', space='cosinesimil_sparse', 
                            data_type=nmslib.DataType.SPARSE_VECTOR)
        self.clusterindex.loadIndex(self.trainargs['cluster_path'], load_data=True)
        if 'query_params' in self.trainargs.keys():
            self.clusterindex.setQueryTimeParams(self.trainargs['query_params'])
