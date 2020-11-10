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
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser("finetuneEBM.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="")
    # fingerprint params
    parser.add_argument("--representation", help="reaction representation", type=str, default="fingerprint")
    # training params 
    parser.add_argument("--model_name", help="model name", type=str, default="FeedforwardEBM")
    parser.add_argument("--old_expt_name", help="old experiment name", type=str, default="")
    parser.add_argument("--expt_name", help="experiment name", type=str, default="")
    parser.add_argument("--precomp_file_prefix",
                        help="precomputed rxn_fp file prefix, expt.py will append f'_{phase}.npz' to the end",
                        type=str, default="")
    parser.add_argument("--date_trained", help="date trained (DD_MM_YYYY)", type=str, default="02_11_2020")
    parser.add_argument("--checkpoint_folder", help="checkpoint folder",
                        type=str, default=expt_utils.setup_paths("LOCAL"))
    parser.add_argument("--batch_size", help="batch_size", type=int, default=2048)
    parser.add_argument("--optimizer", help="optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", help="num. of epochs", type=int, default=30)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-3)
    parser.add_argument("--lr_scheduler", help="learning rate schedule", type=str, default="ReduceLROnPlateau")
    parser.add_argument("--lr_scheduler_factor", help="factor by which learning rate will be reduced", type=float, default=0.3)
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
    if args_type == "train_args":
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


def finetune(args):
    """finetune a trained EBM"""
    logging.info("Setting up model and experiment") 
    train_args = args_to_dict(args, "train_args")

    old_checkpoint_folder = expt_utils.setup_paths(
        "LOCAL", load_trained=True, date_trained=args.date_trained
    )
    saved_stats_filename = f'{args.model_name}_{args.old_expt_name}_stats.pkl'
    saved_model, saved_optimizer, saved_stats = expt_utils.load_model_opt_and_stats(
        saved_stats_filename, old_checkpoint_folder, args.model_name, train_args['optimizer']
    )

    experiment = expt.Experiment(
        model=saved_model,
        model_args=saved_stats["model_args"],
        **train_args,
        **saved_stats["fp_args"],
        augmentations=saved_stats["augmentations"],
        representation=args.representation,
        load_checkpoint=True, 
        saved_optimizer=saved_optimizer,
        saved_stats=saved_stats,
        saved_stats_filename=saved_stats_filename,
        begin_epoch=0,
    )

    logging.info("Start finetuning")
    experiment.train()
    experiment.test()
    experiment.get_topk_acc(phase="train", k=1)
    experiment.get_topk_acc(phase="test", k=1)

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

    finetune(args)
