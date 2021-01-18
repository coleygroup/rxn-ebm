# import shutil
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import get_worker_info


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
    * :code:`Swish`
    * :code:`LearnedSwish`
    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    elif activation == "PReLU":
        return nn.PReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "SELU":
        return nn.SELU()
    elif activation == "ELU":
        return nn.ELU()
    elif activation == 'Swish':
        return Swish()
    elif activation == 'LearnedSwish':
        return LearnedSwish()
    elif activation == 'Mish':
        return Mish()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x) 

class LearnedSwish(nn.Module):
    def __init__(self, slope = 1):
        super().__init__()
        self.slope = slope * torch.nn.Parameter(torch.ones(1))
        self.slope.requiresGrad = True # trainable parameter 
    
    def forward(self, x):
        return self.slope * x * torch.sigmoid(x) 

class Mish(nn.Module):
    '''
    Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681v1
    implemented for PyTorch / FastAI by lessw2020 
    github: https://github.com/lessw2020/mish
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return x *( torch.tanh(F.softplus(x)) )
        

def get_optimizer(optimizer: str) -> torch.optim.Optimizer:
    if optimizer == 'Adam':
        return torch.optim.Adam
    elif optimizer == 'AdamW':
        return torch.optim.AdamW
    elif optimizer == 'Adagrad':
        return torch.optim.Adagrad
    elif optimizer == 'LBFGS':
        return torch.optim.LBFGS
    elif optimizer == 'RMSprop':
        return torch.optim.RMSprop
    elif optimizer == 'SGD':
        return torch.optim.SGD 
    else:
        raise ValueError(f'Optimizer "{optimizer}" not supported.')

def get_lr_scheduler(scheduler: str) -> nn.Module:
    if scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau
    elif scheduler == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    elif scheduler == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR
    else:
        raise ValueError(f'Scheduler "{scheduler}" not supported.')

def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.
    :param model: A PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

def seed_everything(seed: Optional[int] = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    print(f"Using seed: {seed}\n")
