from typing import Optional
import torch
import torch.nn as nn 
torch.backends.cudnn.benchmark = True 

from experiment.expt import Experiment
from experiment.utils import setup_paths, load_model_opt_and_stats
from model.FF import FeedforwardEBM

'''
TODO: in setup_paths(), need to make the necessary folders if they don't exist
i.e. root / 'data' / 'cleaned_data' & root / 'checkpoints' / checkpoint_folder
'''

def trainEBM():
    LOCAL = True # CHANGE THIS  
    expt_name = 'test_precomputed' # CHANGE THIS 

    checkpoint_folder, base_path, cluster_path, sparseFP_vocab_path = setup_paths(expt_name, None, LOCAL)
    precomp_file_prefix = '50k_count_rdm_5' # add f'_{dataset}.npz'

    model_args = {
        'hidden_sizes': [512, 256],  
        'output_size': 1,
        'dropout': 0.1,  
        'activation': 'ReLU'
    }

    fp_args = {
        'rctfp_size': 4096, 
        'prodfp_size': 4096,
        'fp_radius': 3,
        'rxn_type': 'diff',
        'fp_type': 'count'
    }

    train_args = {    
        'batch_size': 64,
        'learning_rate':  5e-3, # to try: lr_finder & lr_schedulers 
        'optimizer': torch.optim.Adam,
        'epochs': 10,
        'early_stop': True,
        'min_delta': 1e-5, 
        'patience': 1,
        'num_workers': 0, # 0 or 4
        'checkpoint': True,
        'random_seed': 0, # affects random sampling & batching
        
        'precomp_file_prefix': precomp_file_prefix,  
        'checkpoint_folder': checkpoint_folder,
        'expt_name': expt_name
    }   

    augmentations = {
        'rdm': {'num_neg': 2}, 
        # 'cos': {'num_neg': 2},
        # 'bit': {'num_neg': 2, 'num_bits': 3}
    }

    model = FeedforwardEBM(**model_args, **fp_args)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        torch.distributed.init_process_group(backend='nccl')
        model = nn.DataParallel(model)
        distributed = True
    else:
        distributed = False

    experiment = Experiment(model, augmentations= augmentations, **train_args, **fp_args, distributed=distributed)
    experiment.train()
    experiment.test()
    scores = experiment.get_topk_acc(key='test', k=1, repeats=1,  name_scores='scores_{}_{}'.format('test', expt_name))
    scores = experiment.get_topk_acc(key='train', k=1, repeats=1, name_scores='scores_{}_{}'.format('train', expt_name))
 

if __name__ == '__main__': 
    trainEBM()
    # TO DO: parse arguments from terminal