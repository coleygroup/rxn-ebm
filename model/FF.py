import torch
import torch.nn as nn

from model.utils import (get_activation_function, initialize_weights)

class FeedforwardEBM(nn.Module):
    '''
    Currently supports feed-forward networks: 
        FF_diff: takes as input a difference FP of fp_size & fp_radius
        FF_sep: takes as input a concatenation of [reactants FP, product FP] 
        
    trainargs: dictionary containing hyperparameters to be optimised, 
    hidden_sizes must be a list e.g. [1024, 512, 256]
    
    To do: bayesian optimisation
    '''
    def __init__(self, trainargs):
        super(FeedforwardEBM, self).__init__()
        self.output_size = trainargs['output_size']
        self.num_layers = len(trainargs['hidden_sizes']) + 1

        if trainargs['fp_type'] == 'sep':
            self.input_dim = trainargs['rctfp_size'] + trainargs['prodfp_size'] # will be rctfp_size + prodfp_size for FF_sep
        elif trainargs['fp_type'] == 'diff':
            self.input_dim = trainargs['rctfp_size']
            assert trainargs['rctfp_size'] == trainargs['prodfp_size'], 'rctfp_size != prodfp_size, unable to make difference FPs!!!'

        self.create_ffn(trainargs)
        initialize_weights(self)  # is it necessary to initialize weights?? 
    
    def create_ffn(self, trainargs):
        '''
        Creates feed-forward network using trainargs dict
        '''
        dropout = nn.Dropout(trainargs['dropout'])
        activation = get_activation_function(trainargs['activation'])

        if self.num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(self.input_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(self.input_dim, trainargs['hidden_sizes'][0])
            ]
            
            # intermediate hidden layers 
            for i, layer in enumerate(range(self.num_layers - 2)):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(trainargs['hidden_sizes'][i], trainargs['hidden_sizes'][i+1]),
                ])
                
            # last hidden layer
            ffn.extend([
                activation,
                dropout,
                nn.Linear(trainargs['hidden_sizes'][-1], self.output_size),
            ])

        self.ffn = nn.Sequential(*ffn)
        
    def forward(self, batch):
        '''
        Runs FF_ebm on input
        
        batch: a N x K x 1 tensor of N training samples, where each sample contains 
        a positive rxn on the first column, and K-1 negative rxn on subsequent columns 
        supplied by DataLoader on custom ReactionDataset 
        '''
        energy_scores = self.ffn(batch) # tensor of size N x K x 1
        return energy_scores.squeeze(dim=-1)  # scores: N x K after squeezing