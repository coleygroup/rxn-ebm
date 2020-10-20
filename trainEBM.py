from data import dataset
from model import FF  
from experiment import utils  
from experiment import expt 
from typing import Optional
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True

def trainEBM():
    expt_name = 'rdm_0_cos_0_bit_10_1_2'  # CHANGE THIS
    # CHANGE THIS, expt.py will add f'_{dataset}.npz'
    precomp_file_prefix = '50k_' + expt_name

    augmentations = {
        'rdm': {'num_neg': 0},
        'cos': {'num_neg': 0},
        'bit': {'num_neg': 10, 'num_bits': 1, 'increment_bits': 2}
    }
    ### PRECOMPUTE ###
    lookup_dict_filename = '50k_mol_smi_to_sparse_fp_idx.pickle'
    mol_fps_filename = '50k_count_mol_fps.npz'
    search_index_filename = '50k_cosine_count.bin'
    augmented_data = dataset.AugmentedData(
        augmentations,
        lookup_dict_filename,
        mol_fps_filename,
        search_index_filename,
        num_workers=8)

    rxn_smis_file_prefix = '50k_clean_rxnsmi_noreagent'
    for dataset in ['train', 'valid', 'test']:
        augmented_data.precompute(
            output_filename=precomp_file_prefix + f'_{dataset}.npz',
            rxn_smis=rxn_smis_file_prefix + f'_{dataset}.pickle',
            distributed=False, parallel=False)

    checkpoint_folder = utils.setup_paths('LOCAL')
    model_args = {
        'hidden_sizes': [1024, 128],
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
        'batch_size': 4096,
        'learning_rate': 6e-3,  # to try: lr_finder & lr_schedulers
        'optimizer': torch.optim.Adam,
        'epochs': 30,
        'early_stop': True,
        'min_delta': 1e-4,
        'patience': 2,
        'num_workers': 0,  # 0 to 8
        'checkpoint': True,
        'random_seed': 0,  # affects RandomAugmentor's sampling & DataLoader's sampling

        'precomp_file_prefix': precomp_file_prefix,
        'checkpoint_folder': checkpoint_folder,
        'expt_name': expt_name
    }

    model = FF.FeedforwardFingerprint(**model_args, **fp_args)

    # if torch.cuda.device_count() > 1:
    #     print('Using {} GPUs'.format(torch.cuda.device_count()))
    #     torch.distributed.init_process_group(backend='nccl')
    #     model = nn.DataParallel(model)
    #     distributed = True
    # else:
    #     distributed = False
    experiment = expt.Experiment(
        model,
        model_args,
        augmentations=augmentations,
        **train_args,
        **fp_args,
        distributed=False)
    experiment.train()
    experiment.test()
    scores_test = experiment.get_topk_acc(dataset_name='test', k=1, repeats=1)
    scores_train = experiment.get_topk_acc(dataset_name='train', k=1, repeats=1)


def resumeEBM():
    expt_name = 'rdm_0_cos_0_bit_5_3'  # CHANGE THIS
    # CHANGE THIS, expt.py will add f'_{dataset}.npz'
    precomp_file_prefix = '50k_' + expt_name

    augmentations = {
        'rdm': {'num_neg': 0},
        'cos': {'num_neg': 0},
        'bit': {'num_neg': 5, 'num_bits': 3}
    }
    ### PRECOMPUTE ###
    lookup_dict_filename = '50k_mol_smi_to_sparse_fp_idx.pickle'
    mol_fps_filename = '50k_count_mol_fps.npz'
    search_index_filename = '50k_cosine_count.bin'
    augmented_data = dataset.AugmentedData(
        augmentations,
        lookup_dict_filename,
        mol_fps_filename,
        search_index_filename)

    rxn_smis_file_prefix = '50k_clean_rxnsmi_noreagent'
    for dataset in ['train', 'valid', 'test']:
        augmented_data.precompute(
            output_filename=precomp_file_prefix + f'_{dataset}.npz',
            rxn_smis=rxn_smis_file_prefix + f'_{dataset}.pickle')

    load_trained = True
    optimizer_name = 'Adam'
    model_name = 'FeedforwardEBM'
    date_trained = '01_10_2020'
    # f'{model_name}_{old_expt_name}_stats.pkl'
    saved_stats_filename = 'FeedforwardEBM_rdm_1_cos_1_bit_1_5_stats.pkl'

    checkpoint_folder = utils.setup_paths('LOCAL', load_trained, date_trained)
    saved_model, saved_optimizer, saved_stats = utils.load_model_opt_and_stats(
                                                    saved_stats_filename, 
                                                    checkpoint_folder, 
                                                    model_name, 
                                                    optimizer_name)

    # if using all same stats as before, just use: saved_stats['model_args'], ['train_args'], ['fp_args']
    # as parameters into Experiment (with **dictionary unpacking), otherwise,
    # define again below
    train_args = {
        'batch_size': 4096,
        'learning_rate': 5e-3,  # to try: lr_finder & lr_schedulers
        'optimizer': torch.optim.Adam,
        'epochs': 5,
        'early_stop': True,
        'min_delta': 1e-4,
        'patience': 1,
        'num_workers': 0,  # 0 to 8
        'checkpoint': True,
        'random_seed': 0,  # affects RandomAugmentor's sampling & DataLoader's sampling

        'precomp_file_prefix': precomp_file_prefix,
        'checkpoint_folder': checkpoint_folder,
        'expt_name': expt_name
    }
    experiment = expt.Experiment(
        saved_model,
        saved_stats['model_args'],
        augmentations=augmentations,
        **train_args,
        **saved_stats['fp_args'],
        load_checkpoint=load_trained,
        saved_optimizer=saved_optimizer,
        saved_stats=saved_stats,
        saved_stats_filename=saved_stats_filename,
        begin_epoch=saved_stats['best_epoch'] + 1)
    experiment.train()
    experiment.test()
    scores_test = experiment.get_topk_acc(dataset_name='test', k=1, repeats=1)
    scores_train = experiment.get_topk_acc(dataset_name='train', k=1, repeats=1)


if __name__ == '__main__':
    trainEBM()
    # resumeEBM()

    # TODO: parse arguments from terminal
