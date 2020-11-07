import argparse
import torch

from rxnebm.data import dataset
from rxnebm.experiment import expt, expt_utils
from rxnebm.model import FF
from rxnebm.model.FF_args import FF_args

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser("trainEBM.py")
    # file names
    parser.add_argument("--smi_to_fp_dict_filename", help="do not change", type=str,
                        default="50k_mol_smi_to_sparse_fp_idx.pickle")
    parser.add_argument("--fp_to_smi_dict_filename", help="do not change", type=str,
                        default="50k_sparse_fp_idx_to_mol_smi.pickle")
    parser.add_argument("--smi_to_fp_dict_filename", help="do not change", type=str,
                        default="50k_mol_smi_to_sparse_fp_idx.pickle")
    parser.add_argument("--mol_fps_filename", help="do not change", type=str,
                        default="50k_count_mol_fps.npz")
    parser.add_argument("--mut_smis_filename", help="do not change", type=str,
                        default="50k_neg150_rad2_maxsize3_mutprodsmis.pickle")
    parser.add_argument("--rxn_smis_file_prefix", help="do not change", type=str,
                        default="50k_clean_rxnsmi_noreagent")
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
    parser.add_argument("--learning_rate", help="learning rate", type=int, default=5e-3)
    parser.add_argument("--optimizer", help="optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", help="num. of epochs", type=int, default=30)
    parser.add_argument("--early_stop", help="whether to use early stopping", action="store_true")
    parser.add_argument("--min_delta", help="what is this", type=float, default=1e-4)
    parser.add_argument("--patience", help="what is this", type=int, default=2)
    parser.add_argument("--num_workers", help="number of workers (0 to 8)", type=int, default=0)
    parser.add_argument("--checkpoint", help="what is this", action="store_true")
    # model params, for now just use model_args with different models

    return parser.parse_args()


def args_to_dict(args, args_type: str) -> dict:
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
                "learning_rate",
                "optimizer",
                "epochs",
                "early_stop",
                "min_delta",
                "patience",
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
        "rdm": {"num_neg": 5},  
        "cos": {"num_neg": 5, "query_params": None},
        "bit": {"num_neg": 5, "num_bits": 1, "increment_bits": 1},
        "mut": {"num_neg": 10},
    }

    # precompute augmentation
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
            distributed=False,
            parallel=False,
        )

    # training
    model_args = FF_args
    fp_args = args_to_dict(args, "fp_args")
    train_args = args_to_dict(args, "train_args")

    model = FF.FeedforwardFingerprint(**model_args, **fp_args)

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


if __name__ == "__main__":
    args = parse_args()

    if not args.resume:
        train(args)
    else:
        raise ValueError(f"Please train from scratch")
