import torch
import torch.tensor as tensor
from torch.utils.data import Dataset

from pathlib import Path
import pickle
import scipy
from scipy import sparse 
import numpy as np
from typing import Optional, Union

from data.augmentors import Augmentor

class ReactionDataset(Dataset): 
    ''' 
    NOTE: ReactionDataset assumes that rxn_fp already exists, 
    unless onthefly = True, in which case, an already initialised augmentor object must be passed.
    Otherwise, an error is raised and training is interrupted. 
    NOTE: experiment.py will coordinate rxn_fp_filename (appending train, valid, test) & .npz
    TODO: coordinate on the fly augmentation (do it at experiment.py)
    TODO: viz_neg: visualise cosine negatives (trace index back to CosineAugmentor & 50k_rxnsmi)

    Parameters
    ----------
    onthefly : bool (Default = False)
        whether to generate augmentations on the fly
        if pre-computed filename is given, loading that file takes priority
    '''
    def __init__(self, input_dim: int, precomp_rxnfp_filename: Optional[str]=None, 
                rxn_smis_filename: Optional[str]=None,
                onthefly: bool=False, augmentor: Optional[Augmentor]=None,
                root: Optional[str]=None, viz_neg: Optional[bool]=False):
        self.input_dim = input_dim
        if root is None:
            root = Path(__file__).parents[1] / 'data' / 'cleaned_data'
        if (root / precomp_rxnfp_filename).exists():
            print('Loading pre-computed reaction fingerprints...')
            self.data = sparse.load_npz(root / precomp_rxnfp_filename)
            self.data = self.data.tocsr()
        elif onthefly:
            print('Generating augmentations on the fly...')
            self.data = augmentor
            self.data.load_smis(rxn_smis_filename)
        else:
            raise Exception('Please provide precomp_rxnfp_filename or set onthefly = True!')
            
    def __getitem__(self, idx: Union[int, tensor]) -> tensor:
        ''' Returns 1 training sample: [pos_rxn_fp, neg_rxn_1_fp, ..., neg_rxn_K-1_fp]
        '''
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        return torch.as_tensor(self.data[idx].toarray().reshape(-1, self.input_dim)).float()
        
    def __len__(self):
        return self.data.shape[0]
 