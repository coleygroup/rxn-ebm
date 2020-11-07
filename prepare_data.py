import argparse
import os
from rxnebm.data.preprocess import clean_smiles, prep_crem, prep_nmslib, smi_to_fp


def parse_args():
    """This is directly copied from trainEBM.py"""
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
    clean_smiles.clean_rxn_smis_50k_all_phases(
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
    smi_to_fp.gen_lookup_dicts_from_file()

    prep_nmslib.build_and_save_index()

    prep_crem.gen_crem_negs(
        num_neg=150, max_size=3, radius=2, frag_db_filename="replacements02_sa2.db"
    )

    print("\nSuccessfully prepared required data!")
    print("#" * 50 + "\n\n")


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args)
