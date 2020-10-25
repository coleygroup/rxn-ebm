import os
import random
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import get_worker_info

# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger('nmslib').setLevel(logging.WARNING) # Only log WARNING
# messages and above from nmslib
import nmslib
from model import FF


def setup_paths(location: str = 'LOCAL',
                load_trained: Optional[bool] = False,
                date_trained: Optional[str] = None,
                root: Optional[Union[str,
                                     bytes,
                                     os.PathLike]] = None):
    '''
    Parameters
    ----------
    root : Union[str, bytes, os.PathLike] (Default = None)
        path to the root folder where checkpoints will be stored
        If None, this is set to full/path/to/rxnebm/checkpoints/
    '''
    if load_trained:
        if date_trained is None:
            raise ValueError('Please provide date_trained as DD_MM_YYYY')
    else:
        date_trained = date.today().strftime("%d_%m_%Y")
    if location.upper() == 'LOCAL':
        if root is None:
            root = Path(__file__).resolve().parents[1] / 'checkpoints'
        checkpoint_folder = Path(root) / date_trained
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f'created checkpoint_folder: {checkpoint_folder}')
    elif location.upper() == 'COLAB':
        if root is None:
            root = Path('/content/gdrive/My Drive/rxn_ebm/checkpoints/')
        checkpoint_folder = Path(root) / date_trained
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f'created checkpoint_folder: {checkpoint_folder}')
    elif location.upper() == 'ENGAGING':
        if root is None:
            root = Path(__file__).resolve().parents[1] / 'checkpoints' 
        checkpoint_folder = Path(root) / date_trained
        os.makedirs(checkpoint_folder, exist_ok=True)
        print(f'created checkpoint_folder: {checkpoint_folder}')
    return checkpoint_folder


def load_model_opt_and_stats(saved_stats_filename: Union[str,
                                                         bytes,
                                                         os.PathLike],
                             checkpoint_folder: Union[str,
                                                      bytes,
                                                      os.PathLike],
                             model_name: str = 'FeedforwardFingerprint',
                             optimizer_name: str = 'Adam'):
    '''
    Parameters
    ----------
    saved_stats_filename : Union[str, bytes, os.PathLike]
        filename or pathlike object to the saved stats dictionary (.pkl)
    checkpoint_folder : Union[str, bytes, os.PathLike]
        path to the checkpoint folder containing the .pth.tar file of the saved model & optimizer weights

    TODO: will need to specify cuda:device_id if doing distributed training
    '''
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_stats = torch.load(
        Path(checkpoint_folder) /
        Path(saved_stats_filename),
        map_location=torch.device(curr_device))

    try:
        checkpoint_filename = saved_stats_filename[:-9] + \
            f'checkpoint_{str(saved_stats["best_epoch"]).zfill(4)}.pth.tar'
        checkpoint = torch.load(
            Path(checkpoint_folder) /
            Path(checkpoint_filename),
            map_location=torch.device(curr_device))
        print('loaded checkpoint from best_epoch: ', saved_stats['best_epoch'])

        if model_name == 'FeedforwardFingerprint' or model_name == 'FeedforwardEBM':
            saved_model = FF.FeedforwardFingerprint(
                **saved_stats['model_args'])
        else:
            print('Only FeedforwardFingerprint is supported currently!')
            return

        if optimizer_name == 'Adam':  # override bug in name of optimizer when saving checkpoint
            saved_stats['train_args']['optimizer'] = torch.optim.Adam
        saved_optimizer = saved_stats['train_args']['optimizer'](
            saved_model.parameters(), lr=saved_stats['train_args']['learning_rate'])

        saved_model.load_state_dict(checkpoint['state_dict'])
        saved_optimizer.load_state_dict(checkpoint['optimizer'])

        if torch.cuda.is_available(
        ):  # move optimizer tensors to gpu  https://github.com/pytorch/pytorch/issues/2830
            for state in saved_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    except Exception as e:
        print(e)
        print('best_epoch: {}'.format(saved_stats['best_epoch']))

    return saved_model, saved_optimizer, saved_stats


def _worker_init_fn_nmslib(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2**31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)

    worker_info = get_worker_info()
    dataset = worker_info.dataset

    if dataset.onthefly:
        with dataset.data.cosaugmentor.search_index as index:
            if index is None:
                index = nmslib.init(method='hnsw', space='cosinesimil_sparse',
                                    data_type=nmslib.DataType.SPARSE_VECTOR)
                index.loadIndex(dataset.search_index_path, load_data=True)
                if dataset.query_params:
                    index.setQueryTimeParams(dataset.query_params)
                else:
                    index.setQueryTimeParams({'efSearch': 100})


def _worker_init_fn_default(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2**31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)
