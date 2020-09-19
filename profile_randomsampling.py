def profile_randomsampling():
    # import torch
    from tqdm import tqdm

    # import sys 
    # from pathlib import Path
    # cwd = Path.cwd()
    # sys.path.append(cwd.parents[0])
    # # print(sys.path)
    
    # import os
    # os.chdir('..')
    # # print(os.getcwd())
    # # return

    # from experiment.experiment import Experiment
    from data.data import ReactionDataset
    # from torch.utils.data import DataLoader
    from experiment.utils import setup_paths, load_model_opt_and_stats
    # from model.FF import FeedforwardEBM

    # torch.backends.cudnn.benchmark = True 

    ################

    LOCAL = True # CHANGE THIS 
    mode = 'random_sampling' # bit_corruption or random_sampling or cosine_pysparnn_<num_index> or cosine_spaces
    expt_name = None
    model_name = None

    checkpoint_folder, base_path, cluster_path, sparseFP_vocab_path = setup_paths(expt_name, mode, LOCAL)

    trainargs = {
    'model': model_name,
    'hidden_sizes': [512, 128],  
    'output_size': 1,
    'dropout': 0.1,  
    
    'batch_size': 64,
    'activation': 'ReLU',  
    'optimizer': None,  #torch.optim.Adam
    'learning_rate':  5e-3, # to try: lr_finder & lr_schedulers 
    'epochs': 1,
    'early_stop': True,
    'min_delta': 1e-5, # we just want to watch out for when val_loss increases
    'patience': 1,
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
    # 'num_bits': 4, 
    # 'num_neg': 5,
    
    'base_path': base_path, # refer to top of notebook 
    'checkpoint_path': checkpoint_folder,
    'cluster_path': cluster_path,
    'sparseFP_vocab_path': sparseFP_vocab_path,
    'query_params': {'efSearch': 100}, # for cosine_spaces, good value to hit 96% recall
 
    'expt_name': expt_name,
    'device': "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }   

    train_dataset = ReactionDataset(trainargs['base_path'], 'train', 
                                        trainargs= trainargs, mode=mode, )
    # pin_memory = True if torch.cuda.is_available() else False
    # train_loader = DataLoader(train_dataset, trainargs['batch_size'], 
    #                                    num_workers= trainargs['num_workers'], 
    #                                     shuffle=True, pin_memory=pin_memory)
    for i in tqdm(range(20000)):
        train_dataset.__getitem__(i)
        continue

if __name__ == '__main__': 
    profile_randomsampling()
    # TO DO: parse arguments from terminal