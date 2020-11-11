import argparse
import logging
import os
import sys
import torch
import gc
gc.enable() 

from datetime import datetime
from rdkit import RDLogger
from rxnebm.data import dataset
from rxnebm.experiment import expt, expt_utils
from rxnebm.model import FF
from rxnebm.model.FF_args import FF_args

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser("trainEBM.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="")
    parser.add_argument("--mol_smi_filename", help="do not change", type=str,
                        default="50k_mol_smis.pickle")
    parser.add_argument("--smi_to_fp_dict_filename", help="do not change", type=str,
                        default="50k_mol_smi_to_sparse_fp_idx.pickle")
    parser.add_argument("--fp_to_smi_dict_filename", help="do not change", type=str,
                        default="50k_sparse_fp_idx_to_mol_smi.pickle")
    parser.add_argument("--mol_fps_filename", help="do not change", type=str,
                        default="50k_count_mol_fps.npz")
    parser.add_argument("--search_index_filename", help="do not change", type=str,
                        default="50k_cosine_count.bin")
    parser.add_argument("--mut_smis_filename", help="do not change", type=str,
                        default="50k_neg150_rad2_maxsize3_mutprodsmis.pickle")
    parser.add_argument("--rxn_smis_file_prefix", help="do not change", type=str,
                        default="50k_clean_rxnsmi_noreagent")
    parser.add_argument("--path_to_energies", help="do not change (folder to store array of energy values for train & test data)", type=str)
    # fingerprint params
    parser.add_argument("--representation", help="reaction representation", type=str, default="fingerprint")
    parser.add_argument("--rctfp_size", help="reactant fp size", type=int, default=4096)
    parser.add_argument("--prodfp_size", help="product fp size", type=int, default=4096)
    parser.add_argument("--fp_radius", help="fp radius", type=int, default=3)
    parser.add_argument("--rxn_type", help="aggregation type", type=str, default="diff")
    parser.add_argument("--fp_type", help="fp type", type=str, default="count")
    # training params
    parser.add_argument("--resume", help="Whether to resume or train from scratch", action="store_true")
    parser.add_argument("--expt_name", help="experiment name", type=str, default="")
    parser.add_argument("--precomp_file_prefix",
                        help="precomputed augmentation file prefix, expt.py will append f'_{phase}.npz' to the end",
                        type=str, default="")
    parser.add_argument("--checkpoint_folder", help="checkpoint folder",
                        type=str, default=expt_utils.setup_paths("LOCAL"))
    parser.add_argument("--batch_size", help="batch_size", type=int, default=2048)
    parser.add_argument("--optimizer", help="optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", help="num. of epochs", type=int, default=30)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-3)
    parser.add_argument("--lr_scheduler", help="learning rate schedule", type=str, default="ReduceLROnPlateau")
    parser.add_argument("--lr_scheduler_factor", help="factor by which learning rate will be reduced", type=float, default=0.2)
    parser.add_argument("--lr_scheduler_patience", help="num. of epochs with no improvement after which learning rate will be reduced", type=int, default=1)
    parser.add_argument("--early_stop", help="whether to use early stopping", action="store_true") # type=bool, default=True) 
    parser.add_argument("--early_stop_patience", help="num. of epochs tolerated without improvement in val loss before early stop", type=int, default=2)
    parser.add_argument("--early_stop_min_delta", help="min. improvement in val loss needed to not early stop", type=float, default=1e-4) 
    parser.add_argument("--num_workers", help="num. of workers (0 to 8)", type=int, default=0)
    parser.add_argument("--checkpoint", help="whether to save model checkpoints", action="store_true") # type=bool, default=True) 
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)
    # model params, for now just use model_args with different models

    return parser.parse_args()


def args_to_dict(args, args_type: str) -> dict:
    """parse args into dict, mainly for compatibility with Min Htoo's code"""
    parsed_dict = {}
    if args_type == "fp_args":
        keys = ["representation",
                "rctfp_size",
                "prodfp_size",
                "fp_radius",
                "rxn_type",
                "fp_type"]
    elif args_type == "train_args":
        keys = ["batch_size",
                "optimizer",
                "epochs",
                "learning_rate",
                "lr_scheduler",
                "lr_scheduler_factor",
                "lr_scheduler_patience",
                "early_stop",
                "early_stop_patience",
                "early_stop_min_delta",
                "num_workers",
                "checkpoint",
                "random_seed",
                "precomp_file_prefix",
                "checkpoint_folder",
                "expt_name"]
    else:
        raise ValueError(f"Unsupported args type: {args_type}")

    for key in keys:
        parsed_dict[key] = getattr(args, key)

    return parsed_dict


def train(args):
    """train EBM from scratch"""

    # hard-coded
    augmentations = {
        "rdm": {"num_neg": 1},  
        "cos": {"num_neg": 1, "query_params": None},
        "bit": {"num_neg": 1, "num_bits": 1, "increment_bits": 1},
        "mut": {"num_neg": 1},
    }

    logging.info("Precomputing augmentation")
    augmented_data = dataset.AugmentedDataFingerprints(
        augmentations=augmentations,
        smi_to_fp_dict_filename=args.smi_to_fp_dict_filename,
        fp_to_smi_dict_filename=args.fp_to_smi_dict_filename,
        mol_fps_filename=args.mol_fps_filename,
        search_index_filename=args.search_index_filename,
        mut_smis_filename=args.mut_smis_filename,
        seed=args.random_seed,
    )

    for phase in ["train", "valid", "test"]:
        augmented_data.precompute(
            output_filename=f"{args.precomp_file_prefix}_{phase}.npz",
            rxn_smis=f"{args.rxn_smis_file_prefix}_{phase}.pickle", 
            parallel=False,
        )

    logging.info("Setting up model and experiment")
    model_args = FF_args
    fp_args = args_to_dict(args, "fp_args")
    train_args = args_to_dict(args, "train_args")

    model = FF.FeedforwardFingerprint(**model_args, **fp_args)

    experiment = expt.Experiment(
        model=model,
        model_args=model_args,
        augmentations=augmentations,
        **train_args,
        **fp_args
    )

    logging.info("Start training")
    experiment.train()
    experiment.test()
    
    _, _ = experiment.get_energies_and_loss(phase="train", save_energies=True, path_to_energies=args.path_to_energies)
    _, _ = experiment.get_energies_and_loss(phase="val", save_energies=True, path_to_energies=args.path_to_energies)
    _, _ = experiment.get_energies_and_loss(phase="test", save_energies=True, path_to_energies=args.path_to_energies)
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        experiment.get_topk_acc(phase="train", k=k)
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        experiment.get_topk_acc(phase="val", k=k)
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        experiment.get_topk_acc(phase="test", k=k)

if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.propagate = False
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if not args.resume:
        train(args)
    else:
        raise ValueError(f"Please train from scratch")
