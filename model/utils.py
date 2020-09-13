import shutil
import os
import random
import numpy as np
import torch
import torch.nn as nn

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
