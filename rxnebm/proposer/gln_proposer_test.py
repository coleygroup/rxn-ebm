from gln_proposer import GLNProposer


def test():
    proposer = GLNProposer(model_path="./rxnebm/proposer/GLN/dropbox/schneider50k.ckpt")
    product_smiles = [
        "[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]",
        "[Br:1][c:12]1[cH:11][cH:10][c:9]([OH:15])[c:8]([Cl:7])[c:13]1[Cl:14]"
    ]
    rxn_types = ["UNK", "UNK"]

    assert len(product_smiles) == len(rxn_types)
    results = proposer.propose(product_smiles, rxn_types, topk=5, beam_size=5)
    print([results])


if __name__ == "__main__":
    test()
