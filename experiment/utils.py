from datetime import date
import os
import torch
import random
import numpy as np
import nmslib
from typing import Optional

from model.FF import FeedforwardEBM

def setup_paths(LOCAL: Optional[bool]=True, 
            load_trained: Optional[bool]=False, date_trained: Optional[bool]=None):
    ''' TODO: engaging_cluster directories
    '''
    if load_trained:
        assert date_trained is not None, 'Please provide date_trained as "DD_MM_YYYY"'
    else:
        date_trained = date.today().strftime("%d_%m_%Y")
    if LOCAL: 
        checkpoint_folder = f'checkpoints/{date_trained}/'
        try:
            os.makedirs(checkpoint_folder)
            print(f'created checkpoint_folder: ', checkpoint_folder)
        except:
            print('checkpoint_folder already exists')
    else: # colab, and soon, engaging_cluster 
        checkpoint_folder = f'/content/gdrive/My Drive/rxn_ebm/checkpoints/{date_trained}/' 
        try:
            os.makedirs(checkpoint_folder)
        except:
            print('checkpoint_folder already exists')
    return checkpoint_folder

def load_model_opt_and_stats(stats_filename, base_path, cluster_path,
                         sparseFP_vocab_path, checkpoint_folder, mode,
                         model='Feedforward', opt='Adam'):
    '''
    TODO: will need to specify cuda:device_id if doing distributed training
    '''
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(checkpoint_folder + stats_filename, 
              map_location=torch.device(curr_device))
    stats['args']['checkpoint_folder'] = checkpoint_folder

    # if mode == 'cosine':
    #     stats['trainargs']['cluster_path'] = cluster_path
    #     stats['trainargs']['sparseFP_vocab_path'] = sparseFP_vocab_path

    if opt == 'Adam':
        stats['trainargs']['optimizer'] = torch.optim.Adam # override bug in name of optimizer when saving checkpoint

    try:
        if stats['best_epoch'] is None:
            stats['best_epoch'] = stats['mean_val_loss'].index(stats['min_val_loss'])  
    except: # KeyError 
        stats['best_epoch'] = stats['mean_val_loss'].index(stats['min_val_loss'])  
    stats['trainargs']['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try: 
        checkpoint_filename = stats_filename[:-9]+'checkpoint_{}.pth.tar'.format(str(stats['best_epoch']).zfill(4)) 
        checkpoint = torch.load(checkpoint_folder + checkpoint_filename,
              map_location=torch.device(curr_device))
        print('loaded checkpoint from best_epoch: ', stats['best_epoch'])

        if model=='Feedforward':
            model = FeedforwardEBM(stats['trainargs'])
        else:
            print('Only Feedforward model is supported currently!')
            return
        optimizer = stats['trainargs']['optimizer'](model.parameters(), lr=stats['trainargs']['learning_rate'])

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if torch.cuda.is_available(): # move optimizer tensors to gpu  https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    except Exception as e:
        print(e)
        print('best_epoch: {}'.format(stats['best_epoch']))
    return model, optimizer, stats

def _worker_init_fn_nmslib_(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2**31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)

    worker_info = get_worker_info() 
    dataset = worker_info.dataset 
    if dataset.onthefly:
        bound_search_index = dataset.data.cosaugmentor.search_index
        if bound_search_index is None:
            bound_search_index = nmslib.init(method='hnsw', space='cosinesimil_sparse', 
                                data_type=nmslib.DataType.SPARSE_VECTOR)
            bound_search_index.loadIndex(dataset.search_index_path, load_data=True)
            if dataset.query_params:
                bound_search_index.setQueryTimeParams(dataset.query_params)
            else:
                bound_search_index.setQueryTimeParams({'efSearch': 100})

def _worker_init_fn_default_(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed % (2**31 - 1)
    random.seed(torch_seed)
    np.random.seed(np_seed)