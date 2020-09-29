from datetime import date
import os
import torch
from model.FF import FeedforwardEBM

def setup_paths(expt_name='expt1', mode='random_sampling', 
                LOCAL=True, load_trained=False, date_trained=None):
    ''' TO DO: engaging_cluster directories
    '''
    if load_trained:
        assert date_trained is not None, 'Please provide date_trained as "DD_MM_YYYY"'
    else:
        date_trained = date.today().strftime("%d_%m_%Y")
        
    cluster_path, sparseFP_vocab_path = None, None
    
    if LOCAL: 
        checkpoint_folder = 'checkpoints/{}/'.format(date_trained)
        try:
            os.makedirs(checkpoint_folder)
            print('created checkpoint_folder: ', checkpoint_folder)
        except:
            print('checkpoint_folder already exists')

        if mode == 'bit_corruption':
            base_path = 'USPTO_50k_data/clean_rxn_50k_sparse_rxnFPs'
        else:
            base_path = 'USPTO_50k_data/clean_rxn_50k_sparse_FPs_numrcts'

        if mode == 'cosine':
            cluster_path = 'USPTO_50k_data/spaces_cosinesimil_index.bin'
            sparseFP_vocab_path = 'USPTO_50k_data/50k_all_mols_sparse_FPs.npz'
    else: # colab, and soon, engaging_cluster 
        checkpoint_folder = '/content/gdrive/My Drive/rxn_ebm/checkpoints/{}/'.format(date_trained) 
        try:
            os.makedirs(checkpoint_folder)
        except:
            print('checkpoint_folder already exists')
        if mode == 'bit_corruption':
            base_path = 'content/clean_rxn_50k_sparse_rxnFPs'
        else:
            base_path = 'content/clean_rxn_50k_sparse_FPs_numrcts'
        if mode == 'cosine':
            cluster_path = 'content/spaces_cosinesimil_index.bin'
            sparseFP_vocab_path = 'content/50k_all_mols_sparse_FPs.npz'
    return checkpoint_folder, base_path, cluster_path, sparseFP_vocab_path

def load_model_opt_and_stats(stats_filename, base_path, cluster_path,
                         sparseFP_vocab_path, checkpoint_folder, mode,
                         model='Feedforward', opt='Adam'):
    '''
    TODO: will need to specify cuda:device_id if doing distributed training
    '''
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(checkpoint_folder + stats_filename, 
              map_location=torch.device(curr_device))
    stats['trainargs']['base_path'] = base_path
    stats['trainargs']['checkpoint_path'] = checkpoint_folder

    if mode == 'cosine':
        stats['trainargs']['cluster_path'] = cluster_path
        stats['trainargs']['sparseFP_vocab_path'] = sparseFP_vocab_path

    if opt == 'Adam':
        stats['trainargs']['optimizer'] = torch.optim.Adam # override bug in name of optimizer when saving checkpoint

    try:
        if stats['best_epoch'] is None:
            stats['best_epoch'] = stats['mean_val_loss'].index(stats['min_val_loss']) + 1  # 1-index 
    except: # Key error 
        stats['best_epoch'] = stats['mean_val_loss'].index(stats['min_val_loss']) + 1  # 1-index 
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