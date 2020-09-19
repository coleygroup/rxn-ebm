def trainEBM():
    import torch
    from experiment.experiment import Experiment
    from experiment.utils import setup_paths, load_model_opt_and_stats
    from model.FF import FeedforwardEBM
    torch.backends.cudnn.benchmark = True 

    LOCAL = True # CHANGE THIS 
    mode = 'cosine_spaces' # bit_corruption or random_sampling or cosine_pysparnn_<num_index> or cosine_spaces
    model_name = 'Feedforward'
    expt_name = 'testing_engaging' # CHANGE THIS 

    checkpoint_folder, base_path, cluster_path, sparseFP_vocab_path = setup_paths(expt_name, mode, LOCAL)

    trainargs = {
    'model': model_name,
    'hidden_sizes': [512, 128],  
    'output_size': 1,
    'dropout': 0.1,  
    
    'batch_size': 64,
    'activation': 'ReLU',  
    'optimizer': torch.optim.Adam,
    'learning_rate':  5e-3, # to try: lr_finder & lr_schedulers 
    'epochs': 2,
    'early_stop': True,
    'min_delta': 1e-5, # we just want to watch out for when val_loss increases
    'patience': 1,
    'num_workers': 4, # 0 or 4

    'checkpoint': True,
    'model_seed': 1337,
    'random_seed': 0, # affects neg rxn sampling since it is random
    
    'rctfp_size': 4096, # if fp_type == 'diff', ensure that both rctfp_size & prodfp_size are identical!
    'prodfp_size': 4096,
    'fp_radius': 3,
    'fp_type': 'diff',
    
    'num_neg_prod': 5, 
    'num_neg_rct': 5,
    'num_bits': 4, 
    'num_neg': 5,
    
    'base_path': base_path, # refer to top of notebook 
    'checkpoint_path': checkpoint_folder,
    'cluster_path': cluster_path,
    'sparseFP_vocab_path': sparseFP_vocab_path,
    'query_params': {'efSearch': 100}, # for cosine_spaces, good value to hit 96% recall
    'num_threads': 1, 
 
    'expt_name': expt_name,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }   

    model = FeedforwardEBM(trainargs)
    experiment = Experiment(model, trainargs, mode=mode)

    experiment.train()
    experiment.test()

    scores = experiment.get_topk_acc(key='test', k=1, repeats=1,  name_scores='scores_{}_{}_{}'.format(mode, 'test', expt_name))
    scores = experiment.get_topk_acc(key='train', k=1, repeats=1, name_scores='scores_{}_{}_{}'.format(mode, 'train', expt_name))
 

if __name__ == '__main__': 
    trainEBM()
    # TO DO: parse arguments from terminal