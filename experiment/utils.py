from datetime import date
import os
import torch
from model.FF import FeedforwardEBM

def setup_paths(expt_name='expt1', mode='random_sampling', 
                     load_trained=False, date_trained=None,
                     LOCAL=True, cosine=True, multi=False):
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

        if cosine:
            if mode == 'cosine_spaces':
                cluster_path = 'USPTO_50k_data/spaces_cosinesimil_index.bin'
            elif multi:
                cluster_path = 'USPTO_50k_data/50k_allmols_sparse_FP_clusterIndex.bin'
            else:
                cluster_path = 'USPTO_50k_data/50k_allmols_sparse_FP_MONOclusterIndex.bin'
            sparseFP_vocab_path = 'USPTO_50k_data/50k_all_mols_sparse_FPs.npz'
    else: # colab 
        checkpoint_folder = '/content/gdrive/My Drive/rxn_ebm/checkpoints/{}/'.format(date_trained) 
        try:
            os.makedirs(checkpoint_folder)
        except:
            print('checkpoint_folder already exists')
        if mode == 'bit_corruption':
            base_path = 'content/clean_rxn_50k_sparse_rxnFPs'
        else:
            base_path = 'content/clean_rxn_50k_sparse_FPs_numrcts'
        if cosine:
            if mode == 'cosine_spaces':
                cluster_path = 'content/spaces_cosinesimil_index.bin'
            elif multi:
                cluster_path = 'content/50k_allmols_sparse_FP_clusterIndex.bin'
            else:
                cluster_path = 'content/50k_allmols_sparse_FP_MONOclusterIndex.bin'
            sparseFP_vocab_path = '/content/50k_all_mols_sparse_FPs.npz'
    return checkpoint_folder, base_path, cluster_path, sparseFP_vocab_path

def load_model_opt_and_stats(stats_filename, base_path, cluster_path,
                         sparseFP_vocab_path, checkpoint_folder,
                         model='Feedforward', cosine=False, opt='Adam'):
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = torch.load(checkpoint_folder + stats_filename, 
              map_location=torch.device('cpu'))
    stats['trainargs']['base_path'] = base_path
    stats['trainargs']['checkpoint_path'] = checkpoint_folder

    if cosine:
        stats['trainargs']['cluster_path'] = cluster_path
        stats['trainargs']['sparseFP_vocab_path'] = sparseFP_vocab_path

    if opt == 'Adam':
        stats['trainargs']['optimizer'] = torch.optim.Adam # fix bug in name of optimizer when saving checkpoint

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