import torch
import torch.nn as nn
import torch.tensor as tensor
from typing import List, Optional

from model.utils import get_activation_function, initialize_weights

class FeedforwardFingerprint(nn.Module):
    '''
    Currently supports 2 feedforward models: 
        diff: takes as input a difference FP of fp_size & fp_radius
        sep: takes as input a concatenation of [reactants FP, product FP] 
    
    hidden_sizes : List[int]
        list of hidden layer sizes from layer 0 onwards
        e.g. [1024, 512, 256] = layer 0 has 1024 neurons, layer 1 has 512 neurons etc.
    output_size : Optional[int] (Default = 1)
        how many outputs the model should give. for binary classification, this is just 1
    
    TODO: bayesian optimisation
    '''
    def __init__(self, hidden_sizes: List[int],
                dropout: int, activation: str, 
                output_size: Optional[int]=1,
                rctfp_size: Optional[int]=4096, prodfp_size: Optional[int]=4096, 
                rxn_type: Optional[str]='diff', **kwargs):
        super(FeedforwardFingerprint, self).__init__()
        if rxn_type == 'sep':
            input_dim = rctfp_size + prodfp_size  
        elif rxn_type == 'diff':
            input_dim = rctfp_size
            if rctfp_size != prodfp_size:
                raise ValueError('rctfp_size must equal prodfp_size for difference fingerprints!')

        num_layers = len(hidden_sizes) + 1
        dropout = nn.Dropout(dropout)
        activation = get_activation_function(activation)
        self.build(dropout, activation, hidden_sizes, input_dim, output_size, num_layers)
        initialize_weights(self)  
    
    def __repr__(self):
        return 'FeedforwardEBM' # needed by experiment.py for saving model details
    
    def build(self, dropout: nn.Dropout, activation: nn.Module, 
                hidden_sizes: List, input_dim: int, 
                output_size: int, num_layers: int):
        if num_layers == 1:
            ffn = [
                # dropout,
                nn.Linear(input_dim, output_size)
            ]
        else:
            ffn = [
                # dropout,
                nn.Linear(input_dim, hidden_sizes[0])
            ]
            
            # intermediate hidden layers 
            for i, layer in enumerate(range(num_layers - 2)):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                ])
                
            # last hidden layer
            ffn.extend([
                activation,
                dropout,
                nn.Linear(hidden_sizes[-1], output_size),
            ])
        self.ffn = nn.Sequential(*ffn)
        
    def forward(self, batch: tensor) -> tensor:
        '''
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column, 
            and K-1 negative rxns on all subsequent columns
        '''
        energy_scores = self.ffn(batch) # tensor of size N x K x 1
        return energy_scores.squeeze(dim=-1)  # scores: N x K after squeezing