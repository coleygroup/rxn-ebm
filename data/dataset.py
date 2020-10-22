import os
import pickle
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path
from typing import List, Optional, Union

import nmslib
import numpy as np
import scipy
import torch
import torch.tensor as tensor
from scipy import sparse
from torch.utils.data import Dataset
from tqdm import tqdm

from data import augmentors
from model import model_utils

sparse_fp = scipy.sparse.csr_matrix

class AugmentedData:
    '''
    Parameters
    ----------
    augmentations : dict
        key : str
            name of augmentation
            choose from 'random', 'cosine', 'bit'
        value : dict
            augmentation parameters, where
            key = name of parameter, value = value of that parameter
            e.g. augmentations['bit'] = {'num_neg': 5, 'num_bits': 5, 'strategy': 'default'}
            'random':
                num_neg : int
                    number of negative reactions to generate
            'cosine':
                num_neg : int
                    number of negative reactions to generate
                query_params : dict
                    num_threads : int (Default = 4)
                        number of CPU threads to use for kNN query by nmslib search index
                    efSearch : int (Default = 100)
                        100 is the recommended value to get high recall (96%)
                    **kwargs :
                        see nmslib's setQueryTimeParams documentation for other possible kwargs
            'bit':
                num_neg : int
                    number of negative reactions to generate
                num_bits : int
                    number of bits to corrupt
                strategy : Optional[str]
                    the strategy to corrupt the bits. TODO: implemented soon!!!
    rxn_type : str (Default = 'diff')
        the method to calculate reaction fingerprints
        currently supports 'diff' & 'sep' methods
    fp_type : str (Default = 'count')
        the type of the fingerprints being supplied
        currently supports 'count' & 'bit' fingerprints
    root : str (Default = None)
        full path to the folder containing all the cleaned, input data files, which includes
        lookup_dict, sparse mol_fps, search_index, mut_prod_smis 
        If not provided, aka None, it defaults to full/path/to/rxn-ebm/data/cleaned_data/
    num_workers : int (Default = 0)
        how many workers to parallelize the PyTorch dataloader over
        has implications on whether search_index is loaded immediately or loaded from _worker_init_fn_nmslib_ 
    seed : int (Default = 0)
        random seed to use. affects augmentations which do random selection e.g. Random sampling, Bit corruption, 
        and CReM (in sampling the required number of mutated product SMILES from the pool of all available mutated SMILES)
    '''

    def __init__(
            self,
            augmentations: dict,
            lookup_dict_filename: str,
            mol_fps_filename: str,
            search_index_filename: Optional[str] = None,
            mut_smis_filename: Optional[str] = None,
            rxn_type: Optional[str] = 'diff',
            fp_type: Optional[str] = 'count',
            fp_size: Optional[int] = 4096, 
            radius: Optional[int] = 3, 
            dtype: Optional[str] = 'int16',
            root: Optional[str] = None,
            num_workers: Optional[int] = 0,
            seed: Optional[int] = 0,
            **kwargs):
        model_utils.seed_everything(seed)

        self.lookup_dict_filename = lookup_dict_filename
        self.mol_fps_filename = mol_fps_filename
        self.search_index_filename = search_index_filename
        self.mut_smis_filename = mut_smis_filename
        if root is None:  # set root = path/to/rxn-ebm/
            root = Path(__file__).parents[1] / 'data' / 'cleaned_data'
        self.root = root
        with open(self.root / self.lookup_dict_filename, 'rb') as handle:
            self.lookup_dict = pickle.load(handle)
        self.mol_fps = sparse.load_npz(self.root / self.mol_fps_filename)

        self.rxn_smis = None
        self.fp_type = fp_type
        self.rxn_type = rxn_type
        self.fp_size = fp_size
        self.dtype = dtype
        self.radius = radius
        self.num_workers = num_workers

        self.augs = []  # list of callables: Augmentor.get_one_sample
        for key, value in augmentations.items():
            if value['num_neg'] == 0:
                continue 
            elif key == 'cosine' or key == 'cos' or key == 'neighbor':
                self._init_cosine(**value)
            elif key == 'random' or key == 'rdm':
                self._init_random(**value)
            elif key == 'bit' or key == 'bits' or key == 'fingerprint':
                self._init_bit(**value)
            elif key == 'mutate' or key == 'mut' or key == 'crem':
                self._init_mutate(**value)

    def _init_cosine(self, num_neg: int, query_params: Optional[dict] = None):
        # NOTE: nmslib only accepts str for its filename, not Path objects
        print('Initialising Cosine Augmentor...')
        search_index = None
        if self.num_workers == 0:  # else, load it from _init_worker_fn_nmslib_
            search_index = nmslib.init(
                method='hnsw',
                space='cosinesimil_sparse',
                data_type=nmslib.DataType.SPARSE_VECTOR)
            search_index.loadIndex(
                str(self.root / self.search_index_filename), load_data=True)
            if query_params:
                search_index.setQueryTimeParams(query_params)
            else:  # default efSearch = 100
                search_index.setQueryTimeParams({'efSearch': 100})

        self.cosaugmentor = augmentors.Cosine(
            num_neg,
            self.lookup_dict,
            self.mol_fps,
            search_index,
            self.rxn_type,
            self.fp_type)
        self.augs.append(self.cosaugmentor.get_one_sample)

    def _init_random(self, num_neg: int):
        print('Initialising Random Augmentor...')
        self.rdmaugmentor = augmentors.Random(
            num_neg,
            self.lookup_dict,
            self.mol_fps,
            self.rxn_type,
            self.fp_type)
        self.augs.append(self.rdmaugmentor.get_one_sample)

    def _init_bit(
            self,
            num_neg: int,
            num_bits: int,
            strategy: Optional[str] = None,
            increment_bits: Optional[int] = 1):
        print('Initialising Bit Augmentor...')
        self.bitaugmentor = augmentors.Bit(
            num_neg,
            num_bits,
            increment_bits,
            strategy,
            self.lookup_dict,
            self.mol_fps,
            self.rxn_type,
            self.fp_type)
        if self.fp_type == 'count':
            self.augs.append(self.bitaugmentor.get_one_sample_count)
        elif self.fp_type == 'bit':
            self.augs.append(self.bitaugmentor.get_one_sample_bit)
    
    def _init_mutate(self, num_neg: int):
        print('Initialising Mutate Augmentor...')
        if Path(self.mut_smis_filename).suffix != '.pickle':
            self.mut_smis_filename = str(self.mut_smis_filename) + '.pickle'
        with open(self.root / self.mut_smis_filename, 'rb') as handle:
            mut_smis = pickle.load(handle)

        self.mutaugmentor = augmentors.Mutate(
            num_neg, 
            self.lookup_dict,
            self.mol_fps,
            mut_smis,  
            self.rxn_type,
            self.fp_type,
            self.radius,
            self.fp_size,
            self.dtype)
        self.augs.append(self.mutaugmentor.get_one_sample)

    def get_one_sample(self, rxn_smi: str) -> sparse_fp:
        ''' prepares one sample, which is 1 pos_rxn + K neg_rxns
        where K is the sum of all of the num_neg for each active augmentation
        '''
        rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        pos_rxn_fp = augmentors.make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for aug in self.augs:
            negs = aug(rxn_smi)
            neg_rxn_fps.extend(negs)

        out = sparse.hstack([pos_rxn_fp, *neg_rxn_fps])
        return out  # spy_sparse2torch_sparse(out)

    def __getitem__(self, idx: int) -> sparse_fp:
        ''' Called by ReactionDataset.__getitem__(idx)
        '''
        return self.get_one_sample(self.rxn_smis[idx])

    def precompute_helper(self, query_params: Optional[dict] = None):
        if hasattr(self, 'cosaugmentor'):
            if self.cosaugmentor.search_index is None:
                self.cosaugmentor.search_index = nmslib.init(
                    method='hnsw',
                    space='cosinesimil_sparse',
                    data_type=nmslib.DataType.SPARSE_VECTOR)
                self.cosaugmentor.search_index.loadIndex(
                    str(self.root / self.search_index_filename), load_data=True)
                if query_params:
                    self.cosaugmentor.search_index.setQueryTimeParams(query_params)
                else:
                    self.cosaugmentor.search_index.setQueryTimeParams({'efSearch': 100})

        out = []
        for i in tqdm(range(len(self.rxn_smis))):
            out.append(self[i])
        return out

    def precompute(self,
                   output_filename: str,
                   rxn_smis: Union[List[str],
                                   Union[str,
                                         bytes,
                                         os.PathLike]],
                   distributed: Optional[bool] = False,
                   parallel: Optional[bool] = True,
                   query_params: Optional[dict] = None):
        self.load_smis(rxn_smis)
        if (self.root / output_filename).exists():
            print(self.root / output_filename, 'already exists!')
            return
        else:
            print('Precomputing...')

        if distributed:
            print('distributed computing is not supported now!')
            return
            # from mpi4py import MPI
            # from mpi4py.futures import MPIPoolExecutor as Pool

            # num_workers = MPI.COMM_WORLD.size
            # print(f'Distributing over {num_workers} total workers')
        elif parallel:
            from concurrent.futures import ProcessPoolExecutor as Pool
            try:
                num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                num_workers = os.cpu_count()
            print(f'Parallelizing over {num_workers} cores')
        else:
            from concurrent.futures import ProcessPoolExecutor as Pool
            print('Not parallelizing!')
            num_workers = 1

        with Pool(max_workers=num_workers) as client:
            future = client.submit(self.precompute_helper, query_params)
            diff_fps = future.result()

        diff_fps = sparse.vstack(diff_fps) # COO format
        diff_fps = diff_fps.tocsr() 
        sparse.save_npz(self.root / output_filename, diff_fps)
        return

    def load_smis(
            self, rxn_smis: Union[List[str], Union[str, bytes, os.PathLike]]):
        if isinstance(rxn_smis, list) and isinstance(rxn_smis[0], str):
            print('List of reaction SMILES strings detected.')
            self.rxn_smis = rxn_smis
        elif isinstance(rxn_smis, str):
            print('Loading reaction SMILES from filename provided.')
            with open(self.root / rxn_smis, 'rb') as handle:
                self.rxn_smis = pickle.load(handle)
        else:
            raise ValueError('Error! No reaction SMILES provided.')
        self.shape = (len(self.rxn_smis), self.mol_fps[0].shape[-1])
        # e.g. (40004, 4096) for train, needed to allow .shape[0] attribute
        # from ReactionDataset.__len__()


def spy_sparse2torch_sparse(data: scipy.sparse.csr_matrix) -> tensor:
    """
    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples = data.shape[0]
    features = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    tensor = torch.sparse.IntTensor(
        indices, torch.from_numpy(values), [samples, features])
    return tensor


class ReactionDataset(Dataset):
    '''
    NOTE: ReactionDataset assumes that rxn_fp already exists,
    unless onthefly = True, in which case, an already initialised augmentor object must be passed.
    Otherwise, a RuntimeError is raised and training is interrupted.
    TODO: viz_neg: visualise cosine negatives (trace index back to CosineAugmentor & 50k_rxnsmi)

    Parameters
    ----------
    onthefly : bool (Default = False)
        whether to generate augmentations on the fly
        if pre-computed filename is given, loading that file takes priority
    '''

    def __init__(
            self,
            input_dim: int,
            precomp_rxnfp_filename: str = None,
            onthefly: bool = False,
            rxn_smis_filename: Optional[str] = None,
            augmented_data: Optional[AugmentedData] = None,
            query_params: Optional[dict] = None,
            search_index_filename: Optional[str] = None,
            root: Optional[str] = None,
            viz_neg: Optional[bool] = False):
        self.input_dim = input_dim
        self.onthefly = onthefly  # needed by worker_init_fn
        self.viz_neg = viz_neg  # TODO
        if root is None:
            root = Path(__file__).parents[1] / 'data' / 'cleaned_data'
        if (root / precomp_rxnfp_filename).exists():
            print('Loading pre-computed reaction fingerprints...')
            self.data = sparse.load_npz(root / precomp_rxnfp_filename) 
            self.data = self.data.tocsr() 
        elif self.onthefly:
            print('Generating augmentations on the fly...')
            self.data = augmented_data
            self.data.load_smis(rxn_smis_filename)
            self.query_params = query_params
            self.search_index_path = str(root / search_index_filename)
        else:
            raise RuntimeError(
                'Please provide precomp_rxnfp_filename or set onthefly = True!')

    def __getitem__(self, idx: Union[int, tensor]) -> tensor:
        ''' Returns 1 training sample: [pos_rxn_fp, neg_rxn_1_fp, ..., neg_rxn_K-1_fp]
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.as_tensor(
            self.data[idx].toarray().reshape(-1, self.input_dim)).float()

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    augmentations = {
        'rdm': {'num_neg': 2},
        'cos': {'num_neg': 2},
        'bit': {'num_neg': 2, 'num_bits': 3},
        'mut': {'num_neg': 10},
    }
    lookup_dict_filename = '50k_mol_smi_to_count_fp.pickle'
    mol_fps_filename = '50k_count_mol_fps.npz'
    search_index_filename = '50k_cosine_count.bin'
    augmented_data = AugmentedData(
        augmentations,
        lookup_dict_filename,
        mol_fps_filename,
        search_index_filename)

    rxn_smis_file_prefix = '50k_clean_rxnsmi_noreagent'
    precomp_output_file_prefix = '50k_rdm_5'
    for dataset in ['train', 'valid', 'test']:
        augmented_data.precompute(
            output_filename=precomp_output_file_prefix + f'_{dataset}.npz',
            rxn_smis=rxn_smis_file_prefix + f'_{dataset}.pickle')

    # from tqdm import tqdm
    # with open('data/cleaned_data/50k_clean_rxnsmi_noreagent_train.pickle', 'rb') as handle:
    #     rxnsmi_train = pickle.load(handle)
    # samples = []
    # for i in tqdm(range(len(rxnsmi_train))):
    #     samples.append(augmentor.get_one_sample(rxnsmi_train[i]))
