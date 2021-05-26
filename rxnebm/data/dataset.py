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


def get_features_per_graph_helper_finetune(_args: Tuple[int, List[str]]):
    i, rxn_smiles = _args
    if i % 1000 == 0:
        logging.info(f"Processing {i}th rxn_smi")

    p_smi = rxn_smiles[0].split(">>")[-1]
    r_smis = [rxn_smi.split(">>")[0] for rxn_smi in rxn_smiles]

    minibatch_smiles = [p_smi]
    minibatch_smiles.extend(r_smis)

    return get_features_from_smiles(i, minibatch_smiles)

class ReactionDatasetFingerprints(Dataset):
    """
    Dataset class for fingerprint representation of reactions
    NOTE: ReactionDatasetFingerprints assumes that rxn_fp already exists
    """

    def __init__(
        self,
        input_dim: int,
        precomp_rxnfp_filename: str = None,
        root: Optional[Union[str, bytes, os.PathLike]] = None
    ):
        self.input_dim = input_dim # needed to reshape row vector in self.__getitem__()
        if root is None:
            root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        else:
            root = Path(root)
        if (root / precomp_rxnfp_filename).exists():
            logging.info("Loading pre-computed reaction fingerprints...")
            self.data = sparse.load_npz(root / precomp_rxnfp_filename)
            self.data = self.data.tocsr()
        else:
            raise RuntimeError("Please provide precomp_rxnfp_filename")

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
    """Dataset class for SMILES/Graph representation of reactions, should be good for both GNN and Transformer"""
    def __init__(
        self,
        args,
        phase: str,
        proposals_csv_filename: Optional[str] = None,
        root: Optional[Union[str, bytes, os.PathLike]] = None
    ):
        model_utils.seed_everything(args.random_seed)
        self.args = args
        self.phase = phase
        if root:
            self.root = Path(root)
        else:
            self.root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        self.rxn_smis_filename = proposals_csv_filename
        self.p = None # Pool for multiprocessing

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

    def get_smiles_and_masks(self):
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

    def precompute(self):
        if self.args.representation == 'graph':
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

                helper = get_features_per_graph_helper_finetune
                
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
            # for transformer, preprocessing is light so we generate on the fly
            logging.info(f"No graph features required, computing negatives from scratch")
            self.get_smiles_and_masks()

    def __getitem__(self, idx) -> Tuple[List, List[bool], int, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.args.representation == 'graph':
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

            return minibatch_graph_features, self._masks[idx], idx #, probs
        else:
            return self._rxn_smiles_with_negatives[idx], self._masks[idx], idx #, probs

    def __len__(self):
        return len(self._rxn_smiles_with_negatives)