def resumeEBM():
    import os
    import numpy as np
    from datetime import date
    import torch

    from experiment.experiment import Experiment
    from experiment.utils import load_model_opt_and_stats, setup_paths
    from model.FF import FeedforwardEBM

    LOCAL = True # CHANGE THIS 
    cosine = False # CHANGE THIS 
    multi = False # monocluster or multicluster
    mode = 'bit_corruption' # CHANGE THIS - bit_corruption or random_sampling or cosine
    model = 'Feedforward'
    expt_name = 'expt3'

    load_pretrained = True
    date_trained = '13_09_2020'
    stats_filename = 'Feedforward_expt3_stats.pkl'  
    opt = 'Adam' 

    checkpoint_folder, base_path, cluster_path, sparseFP_vocab_path = setup_paths(expt_name, mode,
                                                                                   load_pretrained, date_trained, 
                                                                                   LOCAL, cosine, multi)

    model, optimizer, stats = load_model_opt_and_stats(stats_filename, base_path, cluster_path,
                                                  sparseFP_vocab_path, checkpoint_folder,
                                                  model, cosine, opt)

    trainargs = {
    'model': model,
    'hidden_sizes': [1024, 256],  
    'output_size': 1,
    'dropout': 0.1,  

    'batch_size': 64,
    'activation': 'ReLU',  
    'optimizer': torch.optim.Adam,
    'learning_rate':  3e-3, # to try: lr_finder & lr_schedulers 
    'epochs': 50,
    'early_stop': True,
    'min_delta': 1e-5, # we just want to watch out for when val_loss increases
    'patience': 3,
    'num_workers': 4,

    'checkpoint': True,
    'model_seed': 1337,
    'random_seed': 0, # affects neg rxn sampling since it is random

    'rctfp_size': 4096, # if fp_type == 'diff', ensure that both rctfp_size & prodfp_size are identical!
    'prodfp_size': 4096,
    'fp_radius': 3,
    'fp_type': 'diff',

    'num_neg_prod': 5, 
    'num_neg_rct': 5,
    'num_bits': 3, 
    'num_neg': 10,

    'base_path': base_path, # refer to top of notebook 
    'checkpoint_path': checkpoint_folder,
    'cluster_path': cluster_path,
    'sparseFP_vocab_path': sparseFP_vocab_path,
    'query_params': {'efSearch': 100}, # good value to hit 96% recall

    'expt_name': expt_name,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }   

    experiment = Experiment(model, stats['trainargs'], mode=mode, 
                            load_optimizer=optimizer, load_checkpoint=True, load_stats=stats, 
                            stats_filename=stats_filename, begin_epoch=stats['best_epoch'] + 1)

    experiment.train()
    experiment.test()

    scores = experiment.get_topk_acc(key='test', k=1, repeats=1,  name_scores='scores_{}_{}_{}'.format(mode, 'test', expt_name))
    scores = experiment.get_topk_acc(key='train', k=1, repeats=1, name_scores='scores_{}_{}_{}'.format(mode, 'train', expt_name))

if __name__ == '__main__': 
    resumeEBM()
    # TO DO: parse arguments from terminal