import csv
import json
import logging
import nmslib
import os
import pickle
import scipy
import time
import torch
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path
from scipy import sparse
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset
from tqdm import tqdm

from rxnebm.data import augmentors
from rxnebm.data.dataset_utils import get_features_per_graph, smi_tokenizer
from rxnebm.model import model_utils

sparse_fp = scipy.sparse.csr_matrix
Tensor = torch.Tensor


def get_features_from_smiles(i: int, minibatch_smiles: List[str]) -> Tuple[np.ndarray, ...]:
    a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, \
        a_features, a_features_lens, b_features, b_features_lens, \
        a_graphs, b_graphs = zip(*(get_features_per_graph(smi, use_rxn_class=False)
                                   for smi in minibatch_smiles))

    a_scopes = np.concatenate(a_scopes, axis=0)
    b_scopes = np.concatenate(b_scopes, axis=0)
    a_features = np.concatenate(a_features, axis=0)
    b_features = np.concatenate(b_features, axis=0)
    a_graphs = np.concatenate(a_graphs, axis=0)
    b_graphs = np.concatenate(b_graphs, axis=0)

    n_smi_per_minibatch = len(minibatch_smiles)
    minibatch_mol_indexes = np.arange(i*n_smi_per_minibatch, (i+1)*n_smi_per_minibatch)

    return a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, a_features, a_features_lens, b_features, b_features_lens, \
        a_graphs, b_graphs, minibatch_mol_indexes


def get_features_per_graph_helper(_args: Tuple[int, List[str]]):
    i, rxn_smiles = _args
    if i % 1000 == 0:
        logging.info(f"Processing {i}th rxn_smi")

    r_smi = rxn_smiles[0].split(">>")[0]
    p_smis = [rxn_smi.split(">>")[-1] for rxn_smi in rxn_smiles]        # TODO: hardcoded, might be buggy for more crem

    minibatch_smiles = [r_smi]
    minibatch_smiles.extend(p_smis)

    return get_features_from_smiles(i, minibatch_smiles)


def get_features_per_graph_helper_finetune(_args: Tuple[int, List[str]]):
    i, rxn_smiles = _args
    # if i > 1600 and i < 1700:
    #     if i % 1 == 0:
    #         logging.info(f"Processing {i}th rxn_smi")      
    # elif i % 100 == 0:
    #     logging.info(f"Processing {i}th rxn_smi")
    if i % 1000 == 0:
        logging.info(f"Processing {i}th rxn_smi")

    p_smi = rxn_smiles[0].split(">>")[-1]
    r_smis = [rxn_smi.split(">>")[0] for rxn_smi in rxn_smiles]

    minibatch_smiles = [p_smi]
    minibatch_smiles.extend(r_smis)

    return get_features_from_smiles(i, minibatch_smiles)


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
        representation: str = "fingerprint",
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
        self.representation = representation

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
            return_type="fp" if self.representation == "fingerprint" else "smi"
        )
        self.augs.append(self.cosaugmentor.get_one_sample)

    def _init_random(self, num_neg: int):
        logging.info("Initialising Random Augmentor...")
        self.rdmaugmentor = augmentors.Random(
            num_neg=num_neg, 
            smi_to_fp_dict=self.smi_to_fp_dict,
            fp_to_smi_dict=self.fp_to_smi_dict,
            mol_fps=self.mol_fps,
            rxn_type=self.rxn_type,
            fp_type=self.fp_type,
            return_type="fp" if self.representation == "fingerprint" else "smi"
        )
        self.augs.append(self.rdmaugmentor.get_one_sample)

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
            return_type="fp" if self.representation == "fingerprint" else "smi"
        ) 
        self.augs.append(self.bitaugmentor.get_one_sample)

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
            return_type="fp" if self.representation == "fingerprint" else "smi"
        )
        self.augs.append(self.mutaugmentor.get_one_sample)

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
                sparse.save_npz(self.root / f"{Path(output_filename).stem}_{i}.npz", diff_fps_stacked)
                diff_fps = [] # reset diff_fps list
                del diff_fps_stacked

        diff_fps_stacked = sparse.vstack(diff_fps)  # last chunk
        diff_fps_stacked = diff_fps_stacked.tocsr(copy=False) 
        sparse.save_npz(self.root / f"{Path(output_filename).stem}_{i}.npz", diff_fps_stacked)
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
            logging.info("distributed processing is not supported!")
            return
            '''TODO: distributed processing
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

class ReactionDatasetFingerprints(Dataset):
    """
    Dataset class for fingerprint representation of reactions

    NOTE: ReactionDatasetFingerprints assumes that rxn_fp already exists,
    unless onthefly = True, in which case, an already initialised augmentor object must be passed.
    Otherwise, a RuntimeError is raised and training is interrupted

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
        prob_filename: Optional[str] = None,
        root: Optional[Union[str, bytes, os.PathLike]] = None
    ):
        self.input_dim = input_dim # needed to reshape row vector in self.__getitem__()
        self.onthefly = onthefly  # needed by worker_init_fn

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
        
        if prob_filename is not '':
            raise NotImplementedError

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

        return rxn_smi_fps.float(), mask, idx   # return idx for retrieving SMILES from rxn_smi_data

    def __len__(self):
        return self.data.shape[0]


class ReactionDatasetSMILES(Dataset):
    """Dataset class for SMILES representation of reactions, should be good for both GNN and Transformer"""
    def __init__(
        self,
        args,
        phase: str,
        augmentations: dict,
        # precomp_rxnsmi_filename: Optional[str],
        rxn_smis_filename: Optional[str] = None,
        proposals_csv_filename: Optional[str] = None,
        onthefly: bool = False,
        prob_filename: Optional[str] = None,
        root: Optional[Union[str, bytes, os.PathLike]] = None
    ):
        model_utils.seed_everything(args.random_seed)

        self.args = args
        self.phase = phase
        if root:
            self.root = Path(root)
        else:
            self.root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"

        # self.precomp_rxnsmi_filename = precomp_rxnsmi_filename
        self.fp_to_smi_dict_filename = args.fp_to_smi_dict_filename
        self.smi_to_fp_dict_filename = args.smi_to_fp_dict_filename
        # self.mol_fps_filename = args.mol_fps_filename
        self.mut_smis_filename = args.mut_smis_filename

        if args.do_pretrain:
            self.rxn_smis_filename = rxn_smis_filename
        else:
            self.rxn_smis_filename = proposals_csv_filename

        self.p = None # Pool for multiprocessing

        if prob_filename:
            self.prob_data = np.load(self.root / prob_filename)
        else:
            self.prob_data = None

        if onthefly:
            if args.do_pretrain:
                logging.info("Generating augmentations on the fly...")
                with open(self.root / rxn_smis_filename, "rb") as handle:
                    self.rxn_smis = pickle.load(handle)
                with open(self.root / self.smi_to_fp_dict_filename, "rb") as handle:
                    self.smi_to_fp_dict = pickle.load(handle)
                with open(self.root / self.fp_to_smi_dict_filename, "rb") as handle:
                    self.fp_to_smi_dict = pickle.load(handle)
                # self.mol_fps = sparse.load_npz(self.root / self.mol_fps_filename)

                self.augs = []  # list of callables: Augmentor.get_one_sample
                for key, value in augmentations.items():
                    if value["num_neg"] == 0:
                        continue
                    elif key == "cos": # cos not needed for now
                        self._init_cosine(**value)
                    elif key == "rdm":
                        self._init_random(**value)
                    elif key == "mut" or key == "crem":
                        self._init_mutate(**value)
                    else:
                        raise ValueError('Invalid augmentation!')
            elif args.do_finetune:
                logging.info(f"Converting {self.rxn_smis_filename} into json (list of list)")
                self.all_smiles = []
                if self.phase == 'train':
                    with open(self.root / self.rxn_smis_filename, "r") as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        for i, row in enumerate(tqdm(csv_reader)):
                            p_smi = row["prod_smi"]
                            r_smi_true = row["true_precursors"]
                            smiles = [f"{r_smi_true}>>{p_smi}"]
                            for j in range(1, self.args.minibatch_size): # go through columns for proposed precursors
                                cand = row[f"neg_precursor_{j}"]
                                if cand == r_smi_true:
                                    continue
                                if cand.isnumeric() and int(cand) == 9999: # actly can break the loop (but need change if else flow)
                                    continue
                                smiles.append(f"{cand}>>{p_smi}")

                            self.all_smiles.append(smiles)
                else:
                    with open(self.root / self.rxn_smis_filename, "r") as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        for i, row in enumerate(tqdm(csv_reader)):
                            p_smi = row["prod_smi"]
                            smiles = []
                            for j in range(1, self.args.minibatch_eval + 1): # go through columns for proposed precursors
                                cand = row[f"cand_precursor_{j}"]
                                if cand.isnumeric() and int(cand) == 9999: # actly can break the loop (but need change if else flow) # cand.isnumeric() and 
                                    continue
                                smiles.append(f"{cand}>>{p_smi}")
                            self.all_smiles.append(smiles)

            self._rxn_smiles_with_negatives = []
            self._masks = []
            self.minibatch_mol_indexes = []
            self.a_scopes, self.a_scopes_indexes = [], []
            self.b_scopes, self.b_scopes_indexes = [], []
            self.a_features, self.a_features_indexes = [], []
            self.b_features, self.b_features_indexes = [], []
            self.a_graphs, self.b_graphs = [], []
            self.precompute()

        else:
            raise NotImplementedError

    def _init_cosine(self, num_neg: int):
        raise NotImplementedError('Cosine Augemntor is not implemented yet for SMILES')

    def _init_random(self, num_neg: int):
        logging.info("Initialising Random Augmentor...")
        self.rdmaugmentor = augmentors.Random(
            num_neg=num_neg,
            smi_to_fp_dict=self.smi_to_fp_dict,
            fp_to_smi_dict=self.fp_to_smi_dict,
            return_type="smi"
        )
        self.augs.append(self.rdmaugmentor.get_one_sample)

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
            return_type="smi"
        )
        self.augs.append(self.mutaugmentor.get_one_sample)

    def get_smiles_and_masks(self):
        if self.args.do_pretrain:
            for rxn_smi in tqdm(self.rxn_smis):
                minibatch_smiles = [rxn_smi]
                for aug in self.augs:
                    neg_rxn_smiles = aug(rxn_smi)
                    minibatch_smiles.extend(neg_rxn_smiles)

                # crem would give empty string
                minibatch_masks = [bool(smi) and len(smi_tokenizer(smi.split(">>")[-1])) > 1
                                   for smi in minibatch_smiles]

                # hardcode, setting "" or single atom product to "CC"
                # seems that "" or single atom product will just make the indexing more difficult
                minibatch_smiles = [smi if smi and len(smi_tokenizer(smi.split(">>")[-1])) > 1 else "CC"
                                    for smi in minibatch_smiles]

                self._rxn_smiles_with_negatives.append(minibatch_smiles)
                self._masks.append(minibatch_masks)

        elif self.args.do_finetune:
            if self.phase == "train":
                minibatch_size = self.args.minibatch_size
            else:
                minibatch_size = self.args.minibatch_eval

            for rxn_smis_with_neg in tqdm(self.all_smiles):
                # minibatch_size limits number of cand_proposals in finetuning
                minibatch_smiles = rxn_smis_with_neg[:minibatch_size]
                # be careful about defining minibatch_size: for valid/test, true rxn not guaranteed
                while len(minibatch_smiles) < minibatch_size:
                    minibatch_smiles.append("")
                minibatch_masks = [bool(smi) for smi in minibatch_smiles]
                # hardcode, setting "" to "CC"
                # seems that "" or single atom product will just make the indexing more difficult
                minibatch_smiles = [smi if smi else "CC" for smi in minibatch_smiles]

                self._rxn_smiles_with_negatives.append(minibatch_smiles)
                self._masks.append(minibatch_masks)

            """
            if self.phase == 'train':
                for rxn_smis_with_neg in tqdm(self.all_smiles):
                    minibatch_smiles = [rxn_smis_with_neg[0]] 

                    for smi in rxn_smis_with_neg[1:]:
                        minibatch_smiles.append(smi)

                        # minibatch_size limits number of cand_proposals in finetuning
                        if len(minibatch_smiles) == self.args.minibatch_size:
                            # single atom reactants seems not likely for reactant augmentation
                            # so a single bool check should suffice
                            minibatch_masks = [bool(smi) for smi in minibatch_smiles]
                            self._rxn_smiles_with_negatives.append(minibatch_smiles)
                            self._masks.append(minibatch_masks)
                            minibatch_smiles = [rxn_smis_with_neg[0]]       # reset to just 1 element
                            break

                    # pad last minibatch
                    if len(minibatch_smiles) == 1:      # means len(minibatch_smiles) == self.args.minibatch_size
                        continue
                    # be careful about defining minibatch_size: for valid/test, true rxn not guaranteed
                    while len(minibatch_smiles) < self.args.minibatch_size:
                        minibatch_smiles.append("")
                    minibatch_masks = [bool(smi) for smi in minibatch_smiles]
                    minibatch_smiles = [smi if smi else "CC" for smi in minibatch_smiles]

                    self._rxn_smiles_with_negatives.append(minibatch_smiles)
                    self._masks.append(minibatch_masks)
            else:
                for rxn_smis_with_neg in tqdm(self.all_smiles):
                    minibatch_smiles = []   # actly for valid/test, 0th idx means nothing (GT could be of any index)

                    for smi in rxn_smis_with_neg:
                        minibatch_smiles.append(smi)
                        # minibatch_eval limits number of cand_proposals in finetuning
                        if len(minibatch_smiles) == self.args.minibatch_eval:
                            minibatch_masks = [bool(smi) for smi in minibatch_smiles]
                            self._rxn_smiles_with_negatives.append(minibatch_smiles)
                            self._masks.append(minibatch_masks)
                            minibatch_smiles = []       # reset to just 1 element
                            break

                    # pad last minibatch
                    if len(minibatch_smiles) == 0:      # means len(minibatch_smiles) == self.args.minibatch_eval
                        continue

                    # be careful about defining minibatch_size: for valid/test, true rxn not guaranteed
                    while len(minibatch_smiles) < self.args.minibatch_eval:
                        minibatch_smiles.append("")
                    minibatch_masks = [bool(smi) for smi in minibatch_smiles]
                    minibatch_smiles = [smi if smi else "CC" for smi in minibatch_smiles]

                    self._rxn_smiles_with_negatives.append(minibatch_smiles)
                    self._masks.append(minibatch_masks)
            """
        else:
            raise ValueError("Either --do_pretrain or --do_finetune must be supplied!")

    def precompute(self):
        if self.args.do_compute_graph_feat:
            # for graph, we want to cache since the pre-processing is very heavy
            cache_smi = self.root / f"{self.rxn_smis_filename}.{self.args.cache_suffix}.cache_smi.pkl"
            cache_mask = self.root / f"{self.rxn_smis_filename}.{self.args.cache_suffix}.cache_mask.pkl"
            cache_feat = self.root / f"{self.rxn_smis_filename}.{self.args.cache_suffix}.cache_feat.npz"
            cache_feat_index = self.root / f"{self.rxn_smis_filename}.{self.args.cache_suffix}.cache_feat_index.npz"
            if all(os.path.exists(cache) for cache in [cache_smi, cache_mask, cache_feat, cache_feat_index]):
                logging.info(f"Found cache for reaction smiles, masks, features and feature indexes "
                             f"for rxn_smis_fn {self.rxn_smis_filename}, loading")
                with open(cache_smi, "rb") as f:
                    self._rxn_smiles_with_negatives = pickle.load(f)
                with open(cache_mask, "rb") as f:
                    self._masks = pickle.load(f)

                feat = np.load(cache_feat)
                feat_index = np.load(cache_feat_index)
                for attr in ["a_scopes", "b_scopes", "a_features", "b_features", "a_graphs", "b_graphs"]:
                    setattr(self, attr, feat[attr])
                for attr in ["a_scopes", "b_scopes", "a_features", "b_features"]:
                    setattr(self, f"{attr}_indexes", feat_index[f"{attr}_indexes"])
                self.minibatch_mol_indexes = feat_index["minibatch_mol_indexes"]

                logging.info("All loaded.")

            else:
                logging.info(f"Cache not found for rxn_smis_fn {self.rxn_smis_filename} with cache_suffix {self.args.cache_suffix}, \
                            computing from scratch")
                self.get_smiles_and_masks()

                logging.info("Pre-computing graphs and features")
                start = time.time()

                if self.args.do_pretrain:
                    helper = get_features_per_graph_helper
                elif self.args.do_finetune:
                    helper = get_features_per_graph_helper_finetune
                else:
                    raise ValueError("Either --do_pretrain or --do_finetune must be supplied!")

                self.p = Pool(len(os.sched_getaffinity(0)))
                _features_and_lengths = self.p.map(helper, enumerate(self._rxn_smiles_with_negatives))

                a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, \
                    a_features, a_features_lens, b_features, b_features_lens, a_graphs, b_graphs, \
                    minibatch_mol_indexes = zip(*_features_and_lengths)

                self.minibatch_mol_indexes = np.stack(minibatch_mol_indexes, axis=0)

                self.a_scopes = np.concatenate(a_scopes, axis=0)
                self.b_scopes = np.concatenate(b_scopes, axis=0)
                self.a_features = np.concatenate(a_features, axis=0)
                self.b_features = np.concatenate(b_features, axis=0)
                self.a_graphs = np.concatenate(a_graphs, axis=0)
                self.b_graphs = np.concatenate(b_graphs, axis=0)

                def _lengths2indexes(lens):
                    end_indexes = np.cumsum(np.concatenate(lens, axis=0))
                    start_indexes = np.concatenate([[0], end_indexes[:-1]], axis=0)
                    indexes = np.stack([start_indexes, end_indexes], axis=1)
                    return indexes

                self.a_scopes_indexes = _lengths2indexes(a_scopes_lens)
                self.b_scopes_indexes = _lengths2indexes(b_scopes_lens)
                self.a_features_indexes = _lengths2indexes(a_features_lens)
                self.b_features_indexes = _lengths2indexes(b_features_lens)

                logging.info(f"Completed, time: {time.time() - start: .3f} s")
                logging.info(f"Caching...")
                with open(cache_smi, "wb") as of:
                    pickle.dump(self._rxn_smiles_with_negatives, of)
                with open(cache_mask, "wb") as of:
                    pickle.dump(self._masks, of)

                np.savez(cache_feat,
                         a_scopes=self.a_scopes,
                         b_scopes=self.b_scopes,
                         a_features=self.a_features,
                         b_features=self.b_features,
                         a_graphs=self.a_graphs,
                         b_graphs=self.b_graphs)
                np.savez(cache_feat_index,
                         minibatch_mol_indexes=self.minibatch_mol_indexes,
                         a_scopes_indexes=self.a_scopes_indexes,
                         b_scopes_indexes=self.b_scopes_indexes,
                         a_features_indexes=self.a_features_indexes,
                         b_features_indexes=self.b_features_indexes)

                logging.info("All cached.")

                self.p.shutdown(wait=True)          # equivalent to p.close() then p.join()
        else:
            # for transformer, preprocessing is light so we do onthefly
            logging.info(f"No graph features required, computing negatives from scratch")
            self.get_smiles_and_masks()
            #### DEBUGGING ####
            # with open(self.root / f'rxn_smi_with_negs_{self.phase}.pickle', 'wb') as f:
            #     pickle.dump(self._rxn_smiles_with_negatives, f, protocol=4)
            # with open(self.root / f'masks_{self.phase}.pickle', 'wb') as f:
            #     pickle.dump(self._masks, f, protocol=4)

    def __getitem__(self, idx) -> Tuple[List, List[bool], int, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        K = self.args.minibatch_size if self.phase == 'train' else self.args.minibatch_eval 
        if self.prob_data is not None:
            probs = self.prob_data[idx][:K]
        else:
            probs = np.zeros(K)     # for compatibility w/ dataset_utils.collate_fn

        if self.args.do_compute_graph_feat:
            minibatch_mol_index = self.minibatch_mol_indexes[idx]
            minibatch_graph_features = []

            for mol_index in minibatch_mol_index:
                start, end = self.a_scopes_indexes[mol_index]
                a_scope = self.a_scopes[start:end]

                start, end = self.b_scopes_indexes[mol_index]
                b_scope = self.b_scopes[start:end]

                start, end = self.a_features_indexes[mol_index]
                a_feature = self.a_features[start:end]
                a_graph = self.a_graphs[start:end]

                start, end = self.b_features_indexes[mol_index]
                b_feature = self.b_features[start:end]
                b_graph = self.b_graphs[start:end]

                graph_feature = (a_scope, b_scope, a_feature, b_feature, a_graph, b_graph)

                minibatch_graph_features.append(graph_feature)

            return minibatch_graph_features, self._masks[idx], idx, probs
        else:
            return self._rxn_smiles_with_negatives[idx], self._masks[idx], idx, probs

    def __len__(self):
        return len(self._rxn_smiles_with_negatives)

class MixtureDatasetFingerprints(Dataset):
    """
    Dataset class for fingerprint representation of products for predicting which Retro model ('expert')
    could successfully predict the ground truth precursor for each product SMILES
    """

    def __init__(
        self,
        prodfps_filename: str,
        labels_filename: str,
        root: Optional[str] = None,
    ):
        if root is None:
            root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        else:
            root = Path(root)
        if (root / prodfps_filename).exists():
            logging.info("Loading pre-computed product fingerprints...")
            self.data = sparse.load_npz(root / prodfps_filename)
            self.data = self.data.tocsr()
        else:
            raise RuntimeError(
                f"Could not find precomputed product fingerprints at "
                f"{root / prodfps_filename}"
            )

        if (root / labels_filename).exists():
            logging.info("Loading labels...")
            self.labels = np.load(root / labels_filename)
        else:
            raise RuntimeError(
                f"Could not find labels at "
                f"{root / labels_filename}"
            )

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, Tensor, Union[int, List]]:
        """Returns tuple of product fingerprint and index of prod_smi (in CSV file)
        """
        # return idx for retrieving product SMILES & labels from CSV file
        if torch.is_tensor(idx):
            idx = idx.tolist()

        prod_fps = torch.as_tensor(
            self.data[idx].toarray()
        )
        labels = torch.as_tensor(
            self.labels[idx]
        )
        return prod_fps.float(), labels.float(), idx

    def __len__(self):
        return self.data.shape[0]

# if __name__ == "__main__":
    # augmentations = {
    #     "rdm": {"num_neg": 2},
    #     "cos": {"num_neg": 2, "query_params": None},
    #     "bit": {"num_neg": 2, "num_bits": 3, "increment_bits": 1},
    #     "mut": {"num_neg": 10},
    # }

    # smi_to_fp_dict_filename = "50k_mol_smi_to_sparse_fp_idx.pickle"
    # fp_to_smi_dict_filename = "50k_sparse_fp_idx_to_mol_smi.pickle"
    # mol_fps_filename = "50k_count_mol_fps.npz"
    # search_index_filename = "50k_cosine_count.bin"
    # mut_smis_filename = "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"

    # augmented_data = dataset.AugmentedDataFingerprints(
    #     augmentations=augmentations,
    #     smi_to_fp_dict_filename=smi_to_fp_dict_filename,
    #     mol_fps_filename=mol_fps_filename,
    #     search_index_filename=search_index_filename,
    #     mut_smis_filename=mut_smis_filename,
    #     seed=random_seed,
    # )

    # rxn_smis_file_prefix = "50k_clean_rxnsmi_noreagent"
    # for phase in ["train", "valid", "test"]:
    #     augmented_data.precompute(
    #         output_filename=precomp_file_prefix + f"_{phase}.npz",
    #         rxn_smis=rxn_smis_file_prefix + f"_{phase}.pickle",
    #         distributed=False,
    #         parallel=False,
    #     ) 
