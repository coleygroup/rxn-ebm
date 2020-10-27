import argparse
from typing import Optional

import torch
import torch.nn as nn

from data import dataset
from data.preprocess import clean_smiles, smi_to_fp, prep_nmslib, prep_crem
from experiment import expt, expt_utils
from model import FF

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser("trainEBM.py")
    parser.add_argument(
        "--train_from_scratch",
        help="Whether to train from scratch (True) or resume (False)",
        type=bool,
        default=True,
    )

    # file paths
    parser.add_argument(
        "--raw_smi_pre",
        help="File prefix of original raw rxn_smi csv",
        type=str,
        default="schneider50k_raw",
    )
    parser.add_argument(
        "--clean_smi_pre",
        help="File prefix of cleaned rxn_smi pickle",
        type=str,
        default="50k_clean_rxnsmi_noreagent",
    )
    parser.add_argument(
        "--raw_smi_root",
        help="Full path to folder containing raw rxn_smi csv",
        type=str,
    )
    parser.add_argument(
        "--clean_smi_root",
        help="Full path to folder that will contain cleaned rxn_smi pickle",
        type=str,
    )

    # args for clean_rxn_smis_all_phases
    parser.add_argument(
        "--dataset_name",
        help='Name of dataset: "50k", "STEREO" or "FULL"',
        type=str,
        default="50k",
    )
    parser.add_argument(
        "--split_mode",
        help='Whether to keep rxn_smi with multiple products: "single" or "multi"',
        type=str,
        default="multi",
    )
    parser.add_argument(
        "--lines_to_skip", help="Number of lines to skip", type=int, default=1
    )
    parser.add_argument(
        "--keep_reag",
        help="Whether to keep reagents in output SMILES string",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--keep_all_rcts",
        help="Whether to keep all rcts even if they don't contribute atoms to product",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--remove_dup_rxns",
        help="Whether to remove duplicate rxn_smi",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--remove_rct_mapping",
        help="Whether to remove atom map if atom in rct is not in product",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--remove_all_mapping",
        help="Whether to remove all atom map",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--save_idxs",
        help="Whether to save all bad indices to a file in same dir as clean_smi",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--parallelize",
        help="Whether to parallelize computation across all available cpus",
        type=bool,
        default=True,
    )

    # args for get_uniq_mol_smis_all_phases: rxn_smi_file_prefix is same as
    # clean_smi_pre, root is same as clean_smi_root
    parser.add_argument(
        "--mol_smi_filename",
        help="Filename of output pickle file of all unique mol smis",
        type=str,
        default="50k_mol_smis",
    )
    parser.add_argument(
        "--save_reags",
        help="Whether to save unique reagent SMILES strings as separate file",
        type=bool,
        default=False,
    )

    return parser.parse_args()


def prepare_data(args):
    # TODO: parse all arguments
    if args.clean_smi_root:
        print(f"Making dir {args.clean_smi_root}")
        os.makedirs(args.clean_smi_root, exist_ok=True)

    # TODO: add all arguments
    clean_smiles.clean_rxn_smis_all_phases(
        input_file_prefix=args.raw_smi_pre,
        output_file_prefix=args.clean_smi_pre,   
        dataset_name=args.dataset_name,   
        lines_to_skip=args.lines_to_skip,  
        keep_all_rcts=args.keep_all_rcts, 
        remove_dup_rxns=args.remove_dup_rxns, 
        remove_rct_mapping=args.remove_rct_mapping,  
        remove_all_mapping=args.remove_all_mapping,
    )   
    clean_smiles.remove_overlapping_rxn_smis(
        rxn_smi_file_prefix=args.clean_smi_pre,
        root=args.clean_smi_root,
    )
    clean_smiles.get_uniq_mol_smis_all_phases(
        rxn_smi_file_prefix=args.clean_smi_pre,
        root=args.clean_smi_root,
        output_filename=args.mol_smi_filename,
        save_reagents=args.save_reags,
    )

    smi_to_fp.gen_count_mol_fps_from_file()
    smi_to_fp.gen_lookup_dict_from_file()

    prep_nmslib.build_and_save_index()

    prep_crem.gen_crem_negs(
        num_neg=150, max_size=3, radius=2, frag_db_filename="replacements02_sa2.db"
    )

    print("Successfully prepared required data!\n\n")


def trainEBM(args):
    """train EBM from scratch"""

    prepare_data(args)

    expt_name = "50k_rdm_5_cos_5_bit_5_1_1_mut_10"  # USER INPUT
    precomp_file_prefix = "50k_rdm_5_cos_5_bit_5_1_1_mut_10"  # USER INPUT, expt.py will append f'_{phase}.npz' to the end
    random_seed = 0

    augmentations = {  # USER INPUT, pass in 'query_params': dict_of_query_params if desired. see nmslib docs for possible query parameters
        "rdm": {"num_neg": 5},  
        "cos": {"num_neg": 5, "query_params": None},
        "bit": {"num_neg": 5, "num_bits": 1, "increment_bits": 1},
        "mut": {"num_neg": 10},
    }

    #######################################################
    ##################### PRECOMPUTE ######################
    #######################################################
    lookup_dict_filename = "50k_mol_smi_to_sparse_fp_idx.pickle"
    mol_fps_filename = "50k_count_mol_fps.npz"
    search_index_filename = "50k_cosine_count.bin"
    mut_smis_filename = "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"
    augmented_data = dataset.AugmentedData(
        augmentations,
        lookup_dict_filename,
        mol_fps_filename,
        search_index_filename,
        mut_smis_filename,
        seed=random_seed,
    )

    rxn_smis_file_prefix = "50k_clean_rxnsmi_noreagent"
    for phase in ["train", "valid", "test"]:
        augmented_data.precompute(
            output_filename=precomp_file_prefix + f"_{phase}.npz",
            rxn_smis=rxn_smis_file_prefix + f"_{phase}.pickle",
            distributed=False,
            parallel=False,
        )

    #######################################################
    ##################### TRAINING ########################
    #######################################################
    checkpoint_folder = expt_utils.setup_paths("LOCAL")
    model_args = {
        "hidden_sizes": [1024, 512],
        "output_size": 1,
        "dropout": 0.1,
        "activation": "ReLU",
    }

    fp_args = {
        "rctfp_size": 4096,
        "prodfp_size": 4096,
        "fp_radius": 3,
        "rxn_type": "diff",
        "fp_type": "count",
    }

    train_args = {
        "batch_size": 4096,
        "learning_rate": 8e-3,  # to try: lr_finder & lr_schedulers
        "optimizer": "Adam",
        "epochs": 30,
        "early_stop": True,
        "min_delta": 1e-4,
        "patience": 2,
        "num_workers": 0,  # 0 to 8
        "checkpoint": True,
        "random_seed": random_seed,  # affects RandomAugmentor (if onthefly) & DataLoader
        "precomp_file_prefix": precomp_file_prefix,
        "checkpoint_folder": checkpoint_folder,
        "expt_name": expt_name,
    }

    model = FF.FeedforwardFingerprint(**model_args, **fp_args)

    # TODO: add support for multi-GPU training
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
        distributed=False,
    )

    experiment.train()
    experiment.test()
    scores_test = experiment.get_topk_acc(phase="test", k=1)
    scores_train = experiment.get_topk_acc(phase="train", k=1)


def resumeEBM(args):
    """resume training from saved checkpoint. Highly similar to trainEBM() except in
    loading saved model, optimizer & stats (& inferring existing model_args from saved stats file)
    """

    prepare_data(args)

    expt_name = "rdm_0_cos_0_bit_5_3"  # USER INPUT
    precomp_file_prefix = (
        "50k_" + expt_name
    )  # USER INPUT, expt.py will add f'_{phase}.npz'
    augmentations = {  # USER INPUT
        "rdm": {"num_neg": 0},
        "cos": {"num_neg": 0},
        "bit": {"num_neg": 5, "num_bits": 3},
    }

    #######################################################
    ##################### PRECOMPUTE ######################
    #######################################################
    lookup_dict_filename = "50k_mol_smi_to_sparse_fp_idx.pickle"
    mol_fps_filename = "50k_count_mol_fps.npz"
    search_index_filename = "50k_cosine_count.bin"
    mut_smis_filename = "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"
    augmented_data = dataset.AugmentedData(
        augmentations,
        lookup_dict_filename,
        mol_fps_filename,
        search_index_filename,
        mut_smis_filename,
        seed=random_seed,
    )

    rxn_smis_file_prefix = "50k_clean_rxnsmi_noreagent"
    for phase in ["train", "valid", "test"]:
        augmented_data.precompute(
            output_filename=precomp_file_prefix + f"_{phase}.npz",
            rxn_smis=rxn_smis_file_prefix + f"_{phase}.pickle",
            distributed=False,
            parallel=False,
        )

    #######################################################
    ################ LOAD SAVED FILES #####################
    #######################################################
    optimizer_name = "Adam"  # USER INPUT, TODO: infer this from saved_stats
    model_name = "FeedforwardEBM"  # USER INPUT
    date_trained = "01_10_2020"  # USER INPUT
    old_expt_name = "" # USER INPUT 
    saved_stats_filename = f'{model_name}_{old_expt_name}_stats.pkl'
    checkpoint_folder = expt_utils.setup_paths(
        "LOCAL", load_trained=True, date_trained=date_trained
    )
    saved_model, saved_optimizer, saved_stats = expt_utils.load_model_opt_and_stats(
        saved_stats_filename, checkpoint_folder, model_name, optimizer_name
    )

    #######################################################
    ##################### TRAINING ########################
    #######################################################
    # if using all same stats as before, just use: saved_stats['model_args'], ['train_args'], ['fp_args']
    # as parameters into Experiment (with **dictionary unpacking), otherwise, define again below:
    train_args = {
        "batch_size": 4096,
        "learning_rate": 5e-3,
        "optimizer": torch.optim.Adam,
        "epochs": 5,
        "early_stop": True,
        "min_delta": 1e-4,
        "patience": 1,
        "num_workers": 0,
        "checkpoint": True,
        "random_seed": 0,
        "precomp_file_prefix": precomp_file_prefix,
        "checkpoint_folder": checkpoint_folder,
        "expt_name": expt_name,
    }

    experiment = expt.Experiment(
        saved_model,
        saved_stats["model_args"],
        augmentations=augmentations,
        **train_args,
        **saved_stats["fp_args"],
        load_checkpoint=load_trained,
        saved_optimizer=saved_optimizer,
        saved_stats=saved_stats,
        saved_stats_filename=saved_stats_filename,
        begin_epoch=saved_stats["best_epoch"] + 1,
    )

    experiment.train()
    experiment.test()
    scores_test = experiment.get_topk_acc(phase="test", k=1)
    scores_train = experiment.get_topk_acc(phase="train", k=1)


if __name__ == "__main__":
    args = parse_args()

    if args.train_from_scratch:
        trainEBM(args)
    else:
        resumeEBM(args)
