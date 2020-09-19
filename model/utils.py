import shutil
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import get_worker_info

# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger('nmslib').setLevel(logging.WARNING) # Only log WARNING messages and above from nmslib
import nmslib

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
    Supports:
    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`
    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')
    
def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.
    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
            
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def seed_everything(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    print('Using seed: {}'.format(seed))

def _worker_init_fn_nmslib_(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2**31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)

    worker_info = get_worker_info() 
    dataset = worker_info.dataset 
    # print(dataset, dataset.clusterindex)
    if dataset.clusterindex is None:
        dataset.clusterindex = nmslib.init(method='hnsw', space='cosinesimil_sparse', 
                            data_type=nmslib.DataType.SPARSE_VECTOR)
        dataset.clusterindex.loadIndex(dataset.trainargs['cluster_path'], load_data=True)
        if 'query_params' in dataset.trainargs.keys():
            dataset.clusterindex.setQueryTimeParams(dataset.trainargs['query_params'])

def _worker_init_fn_default_(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2**31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)