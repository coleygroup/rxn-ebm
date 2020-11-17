import os
import pickle
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import logging
import scipy
import torch
from scipy import sparse
from torch.utils.data import Dataset
from tqdm import tqdm

import nmslib
from rxnebm.data import augmentors
from rxnebm.model import model_utils

sparse_fp = scipy.sparse.csr_matrix
Tensor = torch.Tensor

class AugmentedDataFingerprints:
    """
    Parameters
    ----------
    augmentations : dict
        key : str
            name of augmentation ['rdm', 'cos', 'bit', 'mut'/'crem']
        value : dict
            augmentation parameters, where
            key = name of parameter, value = value of that parameter
            e.g. augmentations['bit'] = {'num_neg': 5, 'num_bits': 5, 'strategy': 'default'}
            'rdm': (random)
                 num_neg : int
                    number of negative reactions to generate
            'cos': (cosine nearest neighbor search)
                num_neg : int
                    number of negative reactions to generate
                query_params : dict
                    num_threads : int (Default = 4)
                        number of CPU threads to use for kNN query by nmslib search index
                    efSearch : int (Default = 100)
                        100 is the recommended value to get high recall (96%)
                    **kwargs :
                        see nmslib's setQueryTimeParams documentation for other possible kwargs
            'bit': (fingerprint bit corruption)
                num_neg : int
                    number of negative reactions to generate
                num_bits : int
                    number of bits to corrupt
                strategy : Optional[str]
                    the strategy to corrupt the bits. TODO: try this 
            'mut' or 'crem': (CReM aka ContRolled Mutation of Molecules)
                num_neg : int
                    number of negative reactions to generate

    rxn_type : str (Default = 'diff')
        the method to calculate reaction fingerprints
        currently supports 'diff' & 'sep' methods
    fp_type : str (Default = 'count')
        the type of the fingerprints being supplied
        currently supports 'count' & 'bit' fingerprints
    root : str (Default = None)
        full path to the folder containing all the cleaned, input data files, which includes
        smi_to_fp_dict, fp_to_smi_dict, sparse mol_fps, search_index, mut_prod_smis
        If not provided, aka None, it defaults to full/path/to/rxn-ebm/data/cleaned_data/
    seed : int (Default = 0)
        random seed to use. affects augmentations which do random selection e.g. Random sampling, Bit corruption,
        and CReM (in sampling the required number of mutated product SMILES from the pool of all available mutated SMILES)
    """

    def __init__(
        self,
        augmentations: dict,
        smi_to_fp_dict_filename: str,
        fp_to_smi_dict_filename: str,
        mol_fps_filename: str,
        search_index_filename: Optional[str] = None,
        mut_smis_filename: Optional[str] = None,
        rxn_type: Optional[str] = "diff",
        fp_type: Optional[str] = "count",
        fp_size: Optional[int] = 4096,
        radius: Optional[int] = 3,
        dtype: Optional[str] = "int32",
        root: Optional[str] = None,
        seed: Optional[int] = 0,
    ):
        model_utils.seed_everything(seed)

        self.smi_to_fp_dict_filename = smi_to_fp_dict_filename
        self.fp_to_smi_dict_filename = fp_to_smi_dict_filename
        self.mol_fps_filename = mol_fps_filename
        self.search_index_filename = search_index_filename
        self.mut_smis_filename = mut_smis_filename

        if root is None:  # set root = path/to/rxn/ebm/
            root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        else:
            root = Path(root)
        self.root = root

        with open(self.root / self.smi_to_fp_dict_filename, "rb") as handle:
            self.smi_to_fp_dict = pickle.load(handle)
        with open(self.root / self.fp_to_smi_dict_filename, "rb") as handle:
            self.fp_to_smi_dict = pickle.load(handle)
        self.mol_fps = sparse.load_npz(self.root / self.mol_fps_filename)

        self.rxn_smis = None
        self.fp_type = fp_type
        self.rxn_type = rxn_type
        self.fp_size = fp_size
        self.dtype = dtype
        self.radius = radius

        self.augs = []  # list of callables: Augmentor.get_one_sample
        for key, value in augmentations.items():
            if value["num_neg"] == 0:
                continue
            elif key == "cos":
                self._init_cosine(**value)
            elif key == "rdm":
                self._init_random(**value)
            elif key == "bit":
                self._init_bit(**value)
            elif key == "mut" or key == "crem":
                self._init_mutate(**value)
            else:
                raise ValueError('Invalid augmentation!')

    def _init_cosine(self, num_neg: int, query_params: Optional[dict] = None):
        print("Initialising Cosine Augmentor...")
        # loaded later by precompute_helper / expt_utils._worker_init_fn_nmslib_
        search_index = None
        if query_params is not None:
            self.query_params = query_params
        else:
            self.query_params = None

        self.cosaugmentor = augmentors.Cosine(
            num_neg=num_neg, 
            search_index=search_index,
            smi_to_fp_dict=self.smi_to_fp_dict,
            fp_to_smi_dict=self.fp_to_smi_dict,
            mol_fps=self.mol_fps,
            rxn_type=self.rxn_type,
            fp_type=self.fp_type, 
        )
        self.augs.append(self.cosaugmentor.get_one_sample_fp)

    def _init_random(self, num_neg: int):
        logging.info("Initialising Random Augmentor...")
        self.rdmaugmentor = augmentors.Random(
            num_neg=num_neg, 
            smi_to_fp_dict=self.smi_to_fp_dict,
            fp_to_smi_dict=self.fp_to_smi_dict,
            mol_fps=self.mol_fps,
            rxn_type=self.rxn_type,
            fp_type=self.fp_type,    
        )
        self.augs.append(self.rdmaugmentor.get_one_sample_fp)

    def _init_bit(
        self,
        num_neg: int,
        num_bits: int,
        strategy: Optional[str] = None,
        increment_bits: Optional[int] = 1,
    ):
        logging.info("Initialising Bit Augmentor...")
        self.bitaugmentor = augmentors.Bit(
            num_neg=num_neg, 
            num_bits=num_bits,
            increment_bits=increment_bits,
            strategy=strategy,
            smi_to_fp_dict=self.smi_to_fp_dict,
            mol_fps=self.mol_fps,
            rxn_type=self.rxn_type,
            fp_type=self.fp_type,    
        ) 
        self.augs.append(self.bitaugmentor.get_one_sample_fp) 

    def _init_mutate(self, num_neg: int):
        logging.info("Initialising Mutate Augmentor...")
        if Path(self.mut_smis_filename).suffix != ".pickle":
            self.mut_smis_filename = str(self.mut_smis_filename) + ".pickle"
        with open(self.root / self.mut_smis_filename, "rb") as handle:
            mut_smis = pickle.load(handle)

        self.mutaugmentor = augmentors.Mutate(
            num_neg=num_neg,
            mut_smis=mut_smis,
            smi_to_fp_dict=self.smi_to_fp_dict,
            mol_fps=self.mol_fps,
            rxn_type=self.rxn_type,
            fp_type=self.fp_type,
            radius=self.radius,
            fp_size=self.fp_size,
            dtype=self.dtype,
        )
        self.augs.append(self.mutaugmentor.get_one_sample_fp)

    def get_one_minibatch(self, rxn_smi: str) -> sparse_fp:
        """prepares one minibatch of fingerprints: 1 pos_rxn + K neg_rxns
        where K is the sum of all the num_neg for each active augmentation
        """
        rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
        )
        pos_rxn_fp = augmentors.make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        minibatch_neg_rxn_fps = []
        for aug in self.augs:
            neg_rxn_fps = aug(rxn_smi)
            # aug_neg_rxn_smis = aug(rxn_smi)
            # aug_neg_rxn_fps = []
            # for neg_rxn_smi in aug_neg_rxn_smis:
            #     rcts_fp, prod_fp = augmentors.rcts_prod_fps_from_rxn_smi(
            #         neg_rxn_smi, self.fp_type, self.smi_to_fp_dict, self.mol_fps
            #     )
            #     neg_rxn_fp = augmentors.make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)
            #     aug_neg_rxn_fps.append(neg_rxn_fp)
            minibatch_neg_rxn_fps.extend(neg_rxn_fps)

        # TODO: try creating empty sparse vector then allocate elements, see if faster than sparse.hstack
        out = sparse.hstack([pos_rxn_fp, *minibatch_neg_rxn_fps])
        return out  # spy_sparse2torch_sparse(out)


    def __getitem__(self, idx: int) -> sparse_fp:
        """Called by ReactionDatasetFingerprints.__getitem__(idx)"""
        return self.get_one_minibatch(self.rxn_smis[idx])


    def precompute_helper(self):
        if hasattr(self, "cosaugmentor"):
            if self.cosaugmentor.search_index is None:
                self.cosaugmentor.search_index = nmslib.init(
                    method="hnsw",
                    space="cosinesimil_sparse",
                    data_type=nmslib.DataType.SPARSE_VECTOR,
                )
                self.cosaugmentor.search_index.loadIndex(
                    str(self.root / self.search_index_filename), load_data=True
                )
                if self.query_params is not None:
                    self.cosaugmentor.search_index.setQueryTimeParams(self.query_params)
                else:
                    self.cosaugmentor.search_index.setQueryTimeParams({"efSearch": 100})

        out = []
        for i in tqdm(range(len(self.rxn_smis)), desc='Precomputing rxn_fps...'):
            out.append(self[i])
        return out

    def precompute_light_memory(
        self,
        output_filename: str,
        rxn_smis: Union[List[str], Union[str, bytes, os.PathLike]], 
    ):    
        self.load_smis(rxn_smis)

        if (self.root / output_filename).exists():
            logging.info(f"{self.root / output_filename} already exists!")
            return
        else:
            logging.info("Precomputing...")
 
        logging.info("Running light memory version...not parallelizing!")
        if hasattr(self, "cosaugmentor"):
            if self.cosaugmentor.search_index is None:
                self.cosaugmentor.search_index = nmslib.init(
                    method="hnsw",
                    space="cosinesimil_sparse",
                    data_type=nmslib.DataType.SPARSE_VECTOR,
                )
                self.cosaugmentor.search_index.loadIndex(
                    str(self.root / self.search_index_filename), load_data=True
                )
                if self.query_params is not None:
                    self.cosaugmentor.search_index.setQueryTimeParams(self.query_params)
                else:
                    self.cosaugmentor.search_index.setQueryTimeParams({"efSearch": 100})

        diff_fps = []
        for i in tqdm(range(len(self.rxn_smis)), desc='Precomputing rxn_fps...'):
            diff_fps.append(self[i])

            if i > 0 and i % 19000 == 0: # checkpoint
                diff_fps_stacked = sparse.vstack(diff_fps)
                diff_fps_stacked = diff_fps_stacked.tocsr(copy=False) 
                sparse.save_npz(self.root / f"{Path(output_filename).stem()}_{i}.npz", diff_fps_stacked)
                diff_fps = [] # reset diff_fps list
                del diff_fps_stacked

        diff_fps_stacked = sparse.vstack(diff_fps)  # last chunk
        diff_fps_stacked = diff_fps_stacked.tocsr(copy=False) 
        sparse.save_npz(self.root / f"{Path(output_filename).stem()}_{i}.npz", diff_fps_stacked)
        return

    def precompute(
        self,
        output_filename: str,
        rxn_smis: Union[List[str], Union[str, bytes, os.PathLike]],
        distributed: Optional[bool] = False,
        parallel: Optional[bool] = True,
    ):
        self.load_smis(rxn_smis)

        if (self.root / output_filename).exists():
            logging.info(f"{self.root / output_filename} already exists!")
            return
        else:
            logging.info("Precomputing...")

        if distributed:
            logging.info("distributed computing is not supported now!")
            return
            '''TODO: add support & documentation for distributed processing
            '''
            # from mpi4py import MPI
            # from mpi4py.futures import MPIPoolExecutor as Pool

            # num_workers = MPI.COMM_WORLD.size
            # logging.info(f'Distributing over {num_workers} total workers')

            # with Pool(max_workers=num_workers) as client:
                # future = client.submit(self.precompute_helper)
                # diff_fps = future.result()
        elif parallel:
            from concurrent.futures import ProcessPoolExecutor as Pool

            try:
                num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                num_workers = os.cpu_count()
            logging.info(f"Parallelizing over {num_workers} cores")

            with Pool(max_workers=num_workers) as client:
                future = client.submit(self.precompute_helper)
                diff_fps = future.result()
        
        else:
            logging.info("Not parallelizing!")
            if hasattr(self, "cosaugmentor"):
                if self.cosaugmentor.search_index is None:
                    self.cosaugmentor.search_index = nmslib.init(
                        method="hnsw",
                        space="cosinesimil_sparse",
                        data_type=nmslib.DataType.SPARSE_VECTOR,
                    )
                    self.cosaugmentor.search_index.loadIndex(
                        str(self.root / self.search_index_filename), load_data=True
                    )
                    if self.query_params is not None:
                        self.cosaugmentor.search_index.setQueryTimeParams(self.query_params)
                    else:
                        self.cosaugmentor.search_index.setQueryTimeParams({"efSearch": 100})

            diff_fps = []
            for i in tqdm(range(len(self.rxn_smis)), desc='Precomputing rxn_fps...'):
                diff_fps.append(self[i])

        diff_fps_stacked = sparse.vstack(diff_fps, format='csr', dtype='int32')   
        sparse.save_npz(self.root / output_filename, diff_fps_stacked)
        return

    def load_smis(self, rxn_smis: Union[List[str], Union[str, bytes, os.PathLike]]):
        if isinstance(rxn_smis, list) and isinstance(rxn_smis[0], str):
            logging.info("List of reaction SMILES strings detected.\n")
            self.rxn_smis = rxn_smis
        elif isinstance(rxn_smis, str):
            logging.info("Loading reaction SMILES from filename provided.\n")
            with open(self.root / rxn_smis, "rb") as handle:
                self.rxn_smis = pickle.load(handle)
        else:
            raise ValueError("Error! No reaction SMILES provided.")
        self.shape = (len(self.rxn_smis), self.mol_fps[0].shape[-1])
        # e.g. (40004, 4096) for train, needed to allow .shape[0] attribute
        # from ReactionDatasetFingerprints.__len__()

# do not use, is very slow 
# def spy_sparse2torch_sparse(data: scipy.sparse.csr_matrix) -> Tensor:
#     """
#     :param data: a scipy sparse csr matrix
#     :return: a sparse torch tensor
#     """
#     samples = data.shape[0]
#     features = data.shape[1]
#     values = data.data
#     coo_data = data.tocoo()
#     indices = torch.LongTensor([coo_data.row, coo_data.col])
#     tensor = torch.sparse.IntTensor(
#         indices, torch.from_numpy(values), [samples, features]
#     )
#     return tensor


class ReactionDatasetFingerprints(Dataset):
    """
    Dataset class for fingerprint representation of reactions

    NOTE: ReactionDatasetFingerprints assumes that rxn_fp already exists,
    unless onthefly = True, in which case, an already initialised augmentor object must be passed.
    Otherwise, a RuntimeError is raised and training is interrupted.
    TODO: viz_neg: visualise cosine negatives (trace index back to CosineAugmentor & 50k_rxnsmi)

    Parameters
    ----------
    onthefly : bool (Default = False)
        whether to generate augmentations on the fly
        if precomp_rxn_fp_filename is given, loading that file takes priority
    """

    def __init__(
        self,
        input_dim: int,
        precomp_rxnfp_filename: str = None,
        rxn_smis_filename: Optional[str] = None,
        onthefly: bool = False,
        augmented_data: Optional[AugmentedDataFingerprints] = None,
        query_params: Optional[dict] = None,
        search_index_filename: Optional[str] = None,
        proposals_csv_filename: Optional[str] = None, 
        root: Optional[str] = None,
        viz_neg: Optional[bool] = False,
    ):
        self.input_dim = input_dim # needed to reshape row vector in self.__getitem__()
        self.onthefly = onthefly  # needed by worker_init_fn
        self.viz_neg = viz_neg  # TODO

        if root is None:
            root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        else:
            root = Path(root)
        if (root / precomp_rxnfp_filename).exists():
            logging.info("Loading pre-computed reaction fingerprints...")
            self.data = sparse.load_npz(root / precomp_rxnfp_filename)
            self.data = self.data.tocsr()

        elif self.onthefly:
            logging.info("Generating augmentations on the fly...")
            self.data = augmented_data
            self.data.load_smis(rxn_smis_filename)
            # will be used by expt_utils._worker_init_fn_nmslib_
            self.query_params = query_params
            # will be used by expt_utils._worker_init_fn_nmslib_
            self.search_index_path = str(root / search_index_filename)

        else:
            raise RuntimeError(
                "Please provide precomp_rxnfp_filename or set onthefly = True!"
            )

        if proposals_csv_filename is not '' and proposals_csv_filename is not None: # load csv file generated by gen_retrosim_proposals.py, only necessary for valid & test  
            self.proposals_data = pd.read_csv(root / proposals_csv_filename, index_col=None, dtype='str')
            # self.proposals_data.rank_of_true_precursor = self.proposals_data.rank_of_true_precursor.astype('int')
            self.proposals_data = self.proposals_data.drop(['orig_rxn_smi'], axis=1).values 
        else:
            self.proposals_data = None

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, Tensor]:
        """Returns tuple of minibatch & boolean mask

        each minibatch of K rxn fps: [pos_rxn_fp, neg_rxn_1_fp, ..., neg_rxn_K-1_fp]
        if the minibatch is of shape [K, fp_size], the mask is of shape [K]
        the mask is False whenever there is an all-zeros fingerprint vector
        due to insufficient negatives generated from CReM
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rxn_smi_fps = torch.as_tensor(
            self.data[idx].toarray().reshape(-1, self.input_dim)
        )
        mask = torch.sum(rxn_smi_fps.bool(), axis=1).bool()

        return rxn_smi_fps.float(), mask, idx # return idx for retrieving SMILES from rxn_smi_data 

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    augmentations = {
        "rdm": {"num_neg": 2},
        "cos": {"num_neg": 2, "query_params": None},
        "bit": {"num_neg": 2, "num_bits": 3, "increment_bits": 1},
        "mut": {"num_neg": 10},
    }

    smi_to_fp_dict_filename = "50k_mol_smi_to_sparse_fp_idx.pickle"
    fp_to_smi_dict_filename = "50k_sparse_fp_idx_to_mol_smi.pickle"
    mol_fps_filename = "50k_count_mol_fps.npz"
    search_index_filename = "50k_cosine_count.bin"
    mut_smis_filename = "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"

    augmented_data = dataset.AugmentedDataFingerprints(
        augmentations=augmentations,
        smi_to_fp_dict_filename=smi_to_fp_dict_filename,
        mol_fps_filename=mol_fps_filename,
        search_index_filename=search_index_filename,
        mut_smis_filename=mut_smis_filename,
        seed=random_seed,
    )

    rxn_smis_file_prefix = "50k_clean_rxnsmi_noreagent"
    for phase in ["train", "valid", "test"]:
        augmented_data.precompute(
            output_filename=precomp_file_prefix + f"_{phase}.npz",
            rxn_smis=rxn_smis_file_prefix + f"_{phase}.pickle",
            distributed=False,
            parallel=False,
        ) 
