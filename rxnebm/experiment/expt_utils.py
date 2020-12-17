import logging
import os
import random
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import get_worker_info

import nmslib
from rxnebm.model import FF, G2E, S2E, model_utils

def setup_paths(
    location: str = "LOCAL",
    load_trained: Optional[bool] = False,
    date_trained: Optional[str] = None,
    root: Optional[Union[str, bytes, os.PathLike]] = None,
) -> Union[str, bytes, os.PathLike]:
    """ TODO: return #scores_folder #cleaned_data_folder #raw_data_folder
    Parameters
    ----------
    root : Union[str, bytes, os.PathLike] (Default = None)
        path to the root folder where checkpoints will be stored
        If None, this is set to full/path/to/rxnebm/checkpoints/
    """
    if load_trained:
        if date_trained is None:
            raise ValueError("Please provide date_trained as DD_MM_YYYY")
    else:
        date_trained = date.today().strftime("%d_%m_%Y")

    if location.upper() == "LOCAL":
        if root is None:
            root = Path(__file__).resolve().parents[1] / "checkpoints"
        else:
            root = Path(root)
        checkpoint_folder = Path(root) / date_trained
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f"created checkpoint_folder: {checkpoint_folder}")
        # scores_folder, cleaned_data_folder, raw_data_folder = None, None, None

    elif location.upper() == "COLAB":
        if root is None:
            root = Path("/content/gdrive/My Drive/rxn_ebm/checkpoints/")
        else:
            root = Path(root)
        checkpoint_folder = Path(root) / date_trained
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f"created checkpoint_folder: {checkpoint_folder}")

    elif location.upper() == "ENGAGING":
        if root is None:
            root = Path(__file__).resolve().parents[1] / "checkpoints"
        else:
            root = Path(root)
        checkpoint_folder = Path(root) / date_trained
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f"created checkpoint_folder: {checkpoint_folder}")

    return checkpoint_folder #scores_folder #cleaned_data_folder #raw_data_folder


def load_or_create_vocab(args):
    """Currently only supports loading. The vocab is small enough that a single universal vocab suffices"""
    root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"

    vocab = {}
    with open(root / args.vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip()
            vocab[token] = i

    return vocab


def load_model_opt_and_stats(
    args,
    saved_stats_filename: Union[str, bytes, os.PathLike],
    checkpoint_folder: Union[str, bytes, os.PathLike],
    model_name: str = "FeedforwardFingerprint",
    optimizer_name: str = "Adam",
    load_best: bool = True,
    load_epoch : Optional[int] = None
):
    """
    Parameters
    ----------
    saved_stats_filename : Union[str, bytes, os.PathLike]
        filename or pathlike object to the saved stats dictionary (.pkl)
    checkpoint_folder : Union[str, bytes, os.PathLike]
        path to the checkpoint folder containing the .pth.tar file of the saved model & optimizer weights
    load_best : bool (Default = True)
        whether to load the checkpointed model from the best epoch (based on validation loss)
        if false, load_epochs must be provided
    load_epoch : int (Default = None)
        the end of the epoch to load the checkpointed model from

    TODO: will need to specify cuda:device_id if doing distributed training
    """
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_stats = torch.load(
        Path(checkpoint_folder) / Path(saved_stats_filename),
        map_location=torch.device(curr_device),
    )

    try:
        if load_best:
            checkpoint_filename = (
                saved_stats_filename[:-9]
                + f'checkpoint_{str(saved_stats["best_epoch"]).zfill(4)}.pth.tar'
            )
            checkpoint = torch.load(
                Path(checkpoint_folder) / Path(checkpoint_filename),
                map_location=torch.device(curr_device),
            )
            print("loaded checkpoint from best_epoch: ", saved_stats["best_epoch"])
        else:
            checkpoint_filename = (
                saved_stats_filename[:-9]
                + f'checkpoint_{str(load_epoch).zfill(4)}.pth.tar'
            )
            checkpoint = torch.load(
                Path(checkpoint_folder) / Path(checkpoint_filename),
                map_location=torch.device(curr_device),
            )
            print("loaded checkpoint from load_epoch: ", load_epoch)

        if model_name == "FeedforwardFingerprint" or model_name == 'FeedforwardFp' or model_name == "FeedforwardEBM":
            saved_model = FF.FeedforwardSingle(**saved_stats["model_args"], **saved_stats["fp_args"])
        elif model_name == 'FeedforwardTriple3indiv3prod1cos':
            saved_model = FF.FeedforwardTriple3indiv3prod1cos(**saved_stats["model_args"], **saved_stats["fp_args"])
        elif model_name == "GraphEBM":
            saved_model = G2E.G2E(args, **saved_stats["model_args"])
        elif model_name == "TransformerEBM":
            assert args.vocab_file is not None, "Please provide precomputed --vocab_file!"
            vocab = load_or_create_vocab(args)
            saved_model = S2E.S2E(args, vocab, **saved_stats["model_args"])
        else:
            raise ValueError("Only FeedforwardSingle, FeedforwardTriple3indiv3prod1cos, "
                             "GraphEBM and TransformerEBM are supported currently!")

        # override bug in name of optimizer when saving checkpoint
        saved_stats["train_args"]["optimizer"] = model_utils.get_optimizer(optimizer_name)
        saved_optimizer = saved_stats["train_args"]["optimizer"](
            saved_model.parameters(), lr=saved_stats["train_args"]["learning_rate"]
        )

        saved_model.load_state_dict(checkpoint["state_dict"])
        saved_optimizer.load_state_dict(checkpoint["optimizer"])

        if (
            torch.cuda.is_available()
        ):  # move optimizer tensors to gpu  https://github.com/pytorch/pytorch/issues/2830
            for state in saved_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    except Exception as e:
        logging.info(e)
        logging.info("best_epoch: {}".format(saved_stats["best_epoch"]))

    return saved_model, saved_optimizer, saved_stats


def _worker_init_fn_nmslib(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2 ** 31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)

    worker_info = get_worker_info()
    dataset = worker_info.dataset

    if dataset.onthefly:
        with dataset.data.cosaugmentor.search_index as index:
            if index is None:
                index = nmslib.init(
                    method="hnsw",
                    space="cosinesimil_sparse",
                    data_type=nmslib.DataType.SPARSE_VECTOR,
                )
                index.loadIndex(dataset.search_index_path, load_data=True)
                if dataset.query_params:
                    index.setQueryTimeParams(dataset.query_params)
                else:
                    index.setQueryTimeParams({"efSearch": 100})


def _worker_init_fn_default(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2 ** 31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)
