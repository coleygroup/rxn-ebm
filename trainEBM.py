from typing import Optional
import torch
import torch.nn as nn 
torch.backends.cudnn.benchmark = True 

from experiment.expt import Experiment
from experiment.utils import setup_paths, load_model_opt_and_stats
from model.FF import FeedforwardEBM
from data.dataset import AugmentedData
'''
TODO: in setup_paths(), need to make the necessary folders if they don't exist
e.g. root / 'data' / 'cleaned_data' 
root / 'checkpoints' / checkpoint_folder
'''

def trainEBM():
    LOCAL = True # CHANGE THIS  
    expt_name = 'rdm_5_cos_25_bit_35_3' # CHANGE THIS 
    precomp_file_prefix = '50k_' + expt_name # expt.py will add f'_{dataset}.npz'
    
    augmentations = {
        'rdm': {'num_neg': 5}, 
        'cos': {'num_neg': 25},
        'bit': {'num_neg': 35, 'num_bits': 3}
    }
    ### PRECOMPUTE ### 
    lookup_dict_filename = '50k_mol_smi_to_count_fp.pickle'
    mol_fps_filename = '50k_count_mol_fps.npz'
    search_index_filename = '50k_cosine_count.bin'
    augmented_data = AugmentedData(augmentations, lookup_dict_filename, mol_fps_filename, search_index_filename)
    
    rxn_smis_file_prefix = '50k_clean_rxnsmi_noreagent' 
    for dataset in ['train', 'valid', 'test']:
        augmented_data.precompute(
            output_filename=precomp_file_prefix+f'_{dataset}.npz', 
            rxn_smis=rxn_smis_file_prefix+f'_{dataset}.pickle')
    ####################  TODO: fix setup_paths
    checkpoint_folder = setup_paths(LOCAL)
    model_args = {
        'hidden_sizes': [1024, 512, 64],  
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
        'batch_size': 2048,
        'learning_rate':  5e-3, # to try: lr_finder & lr_schedulers 
        'optimizer': torch.optim.Adam,
        'epochs': 30,
        'early_stop': True,
        'min_delta': 1e-6, 
        'patience': 2,
        'num_workers': 0, # 0 to 8
        'checkpoint': True,
        'random_seed': 0, # affects RandomAugmentor's sampling & DataLoader's sampling
        
        'precomp_file_prefix': precomp_file_prefix,  
        'checkpoint_folder': checkpoint_folder,
        'expt_name': expt_name
    }   

    model = FeedforwardEBM(**model_args, **fp_args)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        torch.distributed.init_process_group(backend='nccl')
        model = nn.DataParallel(model)
        distributed = True
    else:
        distributed = False

    experiment = Experiment(model, augmentations=augmentations, **train_args, **fp_args, distributed=distributed)
    experiment.train()
    experiment.test()
    scores = experiment.get_topk_acc(dataset_name='test', k=1, repeats=1)
    scores = experiment.get_topk_acc(dataset_name='train', k=1, repeats=1)
 
if __name__ == '__main__': 
    trainEBM()
    # TO DO: parse arguments from terminal