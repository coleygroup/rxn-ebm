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
    # mode & metadata
    parser.add_argument("--model_name", help="model name", type=str, default="FeedforwardEBM")
    parser.add_argument("--do_pretrain", help="whether to pretrain (vs. finetune)", action="store_true")
    parser.add_argument("--do_finetune", help="whether to finetune (vs. pretrain)", action="store_true")
    parser.add_argument("--do_test", help="whether to test after training", action="store_true")
    parser.add_argument("--do_get_energies_and_acc", help="whether to test after training", action="store_true")
    parser.add_argument("--do_compute_graph_feat", help="whether to compute graph features", action="store_true")
    parser.add_argument("--load_checkpoint", help="whether to load from checkpoint", action="store_true")
    parser.add_argument("--date_trained", help="date trained (DD_MM_YYYY)", type=str, default="02_11_2020")
    parser.add_argument("--expt_name", help="experiment name", type=str, default="")
    parser.add_argument("--old_expt_name", help="old experiment name", type=str, default="")
    parser.add_argument("--checkpoint_folder", help="checkpoint folder",
                        type=str, default=expt_utils.setup_paths("LOCAL"))
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
    parser.add_argument("--path_to_energies",
                        help="do not change (folder to store array of energy values for train & test data)", type=str)
    parser.add_argument("--proposals_csv_file_prefix",
                        help="do not change (CSV file containing proposals from retro models)", type=str,
                        default=None)
    # fingerprint params
    parser.add_argument("--representation", help="reaction representation", type=str, default="fingerprint")
    parser.add_argument("--rctfp_size", help="reactant fp size", type=int, default=4096)
    parser.add_argument("--prodfp_size", help="product fp size", type=int, default=4096)
    parser.add_argument("--fp_radius", help="fp radius", type=int, default=3)
    parser.add_argument("--rxn_type", help="aggregation type", type=str, default="diff")
    parser.add_argument("--fp_type", help="fp type", type=str, default="count")
    # training params
    parser.add_argument("--precomp_file_prefix",
                        help="precomputed augmentation file prefix, expt.py will append f'_{phase}.npz' to the end",
                        type=str, default="")
    parser.add_argument("--batch_size", help="batch_size", type=int, default=2048)
    parser.add_argument("--minibatch_size", help="minibatch size for smiles representation", type=int, default=32)
    parser.add_argument("--optimizer", help="optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", help="num. of epochs", type=int, default=30)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-3)
    parser.add_argument("--lr_scheduler", help="learning rate schedule", type=str, default="ReduceLROnPlateau")
    parser.add_argument("--lr_scheduler_criteria",
                        help="criteria for learning rate scheduler ['loss', 'acc']", type=str, default='acc')
    parser.add_argument("--lr_scheduler_factor",
                        help="factor by which learning rate will be reduced", type=float, default=0.2)
    parser.add_argument("--lr_scheduler_patience",
                        help="num. of epochs with no improvement after which learning rate will be reduced",
                        type=int, default=1)
    parser.add_argument("--early_stop", help="whether to use early stopping", action="store_true") # type=bool, default=True) 
    parser.add_argument("--early_stop_criteria",
                        help="criteria for early stopping ['loss', 'top1_acc', 'top5_acc', 'top10_acc', 'top50_acc']",
                        type=str, default='top1_acc')
    parser.add_argument("--early_stop_patience",
                        help="num. of epochs tolerated without improvement in criteria before early stop",
                        type=int, default=2)
    parser.add_argument("--early_stop_min_delta",
                        help="min. improvement in criteria needed to not early stop", type=float, default=1e-4)
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
    else:
        raise ValueError(f"Unsupported args type: {args_type}")

    for key in keys:
        parsed_dict[key] = getattr(args, key)

    return parsed_dict


def main(args):
    """train EBM from scratch"""
    logging.info("Logging args")
    logging.info(vars(args))

    # hard-coded
    model, model_args, saved_optimizer, saved_stats, saved_stats_filename = None, None, None, None, None
    begin_epoch = 0
    debug = False
    augmentations = {
        "rdm": {"num_neg": 1},  
        "cos": {"num_neg": 1, "query_params": None},
        "bit": {"num_neg": 1, "num_bits": 1, "increment_bits": 1},
        "mut": {"num_neg": 1}
    }

    if args.do_pretrain:
        logging.info("Precomputing augmentation")
        augmented_data = dataset.AugmentedDataFingerprints(
            augmentations=augmentations,
            smi_to_fp_dict_filename=args.smi_to_fp_dict_filename,
            fp_to_smi_dict_filename=args.fp_to_smi_dict_filename,
            mol_fps_filename=args.mol_fps_filename,
            search_index_filename=args.search_index_filename,
            mut_smis_filename=args.mut_smis_filename,
            seed=args.random_seed
        )

        for phase in ["train", "valid", "test"]:
            augmented_data.precompute(
                output_filename=f"{args.precomp_file_prefix}_{phase}.npz",
                rxn_smis=f"{args.rxn_smis_file_prefix}_{phase}.pickle",
                parallel=False
            )

    if args.load_checkpoint:
        logging.info("Loading from checkpoint")
        old_checkpoint_folder = expt_utils.setup_paths(
            "LOCAL", load_trained=True, date_trained=args.date_trained
        )
        saved_stats_filename = f'{args.model_name}_{args.old_expt_name}_stats.pkl'
        saved_model, saved_optimizer, saved_stats = expt_utils.load_model_opt_and_stats(
            saved_stats_filename, old_checkpoint_folder, args.model_name, args.optimizer
        )
        logging.info(f"Saved model {saved_model.model_repr} loaded")
        logging.info("Updating args with fp_args")
        for k, v in saved_stats["fp_args"]:
            setattr(args, k, v)

        model = saved_model
        model_args = saved_stats["model_args"]
        augmentations = saved_stats["augmentations"]
        saved_optimizer = saved_optimizer
        saved_stats = saved_stats
        saved_stats_filename = saved_stats_filename
        begin_epoch = 0
        debug = True
    else:
        logging.info("Not loading from checkpoint, creating model")
        model_args = FF_args
        fp_args = args_to_dict(args, "fp_args")

        model = FF.FeedforwardSingle(**model_args, **fp_args)
        logging.info(f"Model {model.model_repr} created")

    logging.info("Logging model summary")
    logging.info(model)
    logging.info(f"\nModel #Params: {sum([x.nelement() for x in model.parameters()]) / 1000} k")

    logging.info("Setting up experiment")
    experiment = expt.Experiment(
        args=args,
        model=model,
        model_args=model_args,
        augmentations=augmentations,
        load_checkpoint=args.load_checkpoint,
        saved_optimizer=saved_optimizer,
        saved_stats=saved_stats,
        saved_stats_filename=saved_stats_filename,
        begin_epoch=begin_epoch,
        debug=debug
    )

    if args.do_pretrain:
        logging.info("Start pretraining")
        experiment.train()

    if args.do_finetune:
        assert args.proposals_csv_file_prefix is not None, f"Please provide --proposals_csv_file_prefix!"
        logging.info("Start finetuning")
        experiment.train()

    if args.do_test:
        logging.info("Start testing")
        experiment.test()

    if args.do_get_energies_and_acc:
        for phase in ["train", "val", "test"]:
            experiment.get_energies_and_loss(
                phase=phase, finetune=args.do_finetune, save_energies=True, path_to_energies=args.path_to_energies)

        for phase in ["train", "val", "test"]:
            logging.info(f"\nGetting {phase} accuracies")
            for k in [1, 2, 3, 5, 10, 20, 50, 100]:
                experiment.get_topk_acc(phase=phase, k=k)


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

    main(args)
