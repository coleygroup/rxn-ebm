def trainEBM():
    import os
    import numpy as np
    from datetime import date
    import torch

    LOCAL = True # CHANGE THIS 
    cosine = False # CHANGE THIS 
    multi = False # monocluster or multicluster
    mode = None # CHANGE THIS - bit_corruption or None
    expt_name = 'testing_scripts'
    mode = 'random_sampling'

    today = date.today().strftime("%d_%m_%Y")
    if LOCAL: 
        checkpoint_folder = 'checkpoints/{}/'.format(today)
        try:
            os.makedirs(checkpoint_folder)
        except:
            print('checkpoint_folder already exists')
            
        if mode == 'bit_corruption':
            base_path = 'USPTO_50k_data/clean_rxn_50k_sparse_rxnFPs'
        else:
            base_path = 'USPTO_50k_data/clean_rxn_50k_sparse_FPs_numrcts'
        if cosine:
            if multi:
                cluster_path = 'USPTO_50k_data/50k_allmols_sparse_FP_clusterIndex.bin'
            else:
                cluster_path = 'USPTO_50k_data/50k_allmols_sparse_FP_MONOclusterIndex.bin'
            sparseFP_vocab_path = 'USPTO_50k_data/50k_all_mols_sparse_FPs.npz'
    else: # colab 
        checkpoint_folder = '/content/gdrive/My Drive/rxn_ebm/checkpoints/{}/'.format(today) 
        try:
            os.makedirs(checkpoint_folder)
        except:
            print('checkpoint_folder already exists')
        if mode == 'bit_corruption':
            base_path = 'content/clean_rxn_50k_sparse_rxnFPs'
        else:
            base_path = 'content/clean_rxn_50k_sparse_FPs_numrcts'
        if cosine:
            if multi:
                cluster_path = 'content/50k_allmols_sparse_FP_clusterIndex.bin'
            else:
                cluster_path = 'content/50k_allmols_sparse_FP_MONOclusterIndex.bin'
            sparseFP_vocab_path = '/content/50k_all_mols_sparse_FPs.npz'

    from experiment import Experiment
    from model.FF import FeedforwardEBM
    from data.data import ReactionDataset
    from torch.utils.data import DataLoader

    trainargs = {
        'model': 'Feedforward', # must change both model & fp_type 
        'hidden_sizes': [16],  
        'output_size': 1,
        'dropout': 0,  
        
        'batch_size': 64,
        'activation': 'ReLU',  
        'optimizer': torch.optim.Adam,
        'learning_rate': 9e-3, # to try: lr_finder & lr_schedulers 
        'epochs': 2,
        'early_stop': True,
        'min_delta': 1e-4, 
        'patience': 1,

        'checkpoint': True,
        'random_seed': 0, # only 1 seed is needed to seed all (torch, np, random etc.)
        
        'rctfp_size': 4096, # if fp_type == 'diff', ensure that both rctfp_size & prodfp_size are identical!
        'prodfp_size': 4096,
        'fp_radius': 3,
        'fp_type': 'diff',
        
        'num_neg_prod': 1,
        'num_neg_rct': 1,
        
        'base_path': base_path, 
        'checkpoint_path': checkpoint_folder,
        'expt_name': expt_name,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    model = FeedforwardEBM(trainargs)
    experiment = Experiment(model, trainargs, mode=mode)

    experiment.train()
    experiment.test()

    key = 'test'
    dataset = ReactionDataset(trainargs['base_path'], key, trainargs, mode=mode)
    dataloader = DataLoader(dataset, 2 * trainargs['batch_size'], shuffle=False)
    scores = experiment.get_topk_acc(dataloader, k=1, repeats=1, get_loss=True)
    torch.save(scores, 'scores_{}_{}_{}.pkl'.format(mode, key, expt_name)) 
    # fix this naming convention, used by analysis.py 

    key = 'train'
    dataset = ReactionDataset(trainargs['base_path'], key, trainargs, mode=mode)
    dataloader = DataLoader(dataset, 2 * trainargs['batch_size'], shuffle=False)
    scores = experiment.get_topk_acc(dataloader, k=1, repeats=1, get_loss=True)
    torch.save(scores, 'scores_{}_{}_{}.pkl'.format(mode, key, expt_name))
    # fix this naming convention, used by analysis.py 

if __name__ == '__main__': 
    trainEBM()
    # TO DO: parse arguments from terminal