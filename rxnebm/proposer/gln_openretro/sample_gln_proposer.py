import argparse
from models.gln_model.gln_proposer import GLNProposer


def parse_args():
    parser = argparse.ArgumentParser("gln_proposer_sample.py")
    parser.add_argument("--model_name", help="model name", type=str, default="gln")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str,
                        default="./data/gln_schneider50k/processed")
    parser.add_argument("--model_path", help="model output path", type=str,
                        default="./checkpoints/gln_schneider50k/model-6.dump/")

    return parser.parse_args()


def gln_proposer_test(args):
    proposer = GLNProposer(
        model_name=args.model_name,
        model_args=None,
        model_config={},
        data_name="",
        processed_data_path=args.processed_data_path,
        model_path=args.model_path
    )

    product_smiles = [
        "[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]",
        "[Br:1][c:12]1[cH:11][cH:10][c:9]([OH:15])[c:8]([Cl:7])[c:13]1[Cl:14]"
    ]
    rxn_types = ["UNK", "UNK"]

    assert len(product_smiles) == len(rxn_types)
    results = proposer.propose(product_smiles, rxn_types, topk=5, beam_size=50)
    print(results)
    """
    List of n[{"template": List of topk templates,
               "reactants": List of topk reactants,
               "scores": ndarray of topk scores}]
    """


if __name__ == "__main__":
    args = parse_args()
    gln_proposer_test(args)
