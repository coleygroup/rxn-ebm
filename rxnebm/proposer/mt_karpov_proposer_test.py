from rxnebm.proposer.mt_karpov_config import mt_karpov_config
from rxnebm.proposer.mt_karpov_proposer import MTKarpovProposer

def test():
    proposer = MTKarpovProposer(mt_karpov_config)
    # product_smiles = [
    #     "[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]",
    #     "[Br:1][c:12]1[cH:11][cH:10][c:9]([OH:15])[c:8]([Cl:7])[c:13]1[Cl:14]"
    # ]
    # TODO: add in utils for dropping atom map
    product_smiles = [
        "CC(C)(C)OC(=O)NCCc1ccc(N)cc1F",
        "NCC1=CC[C@@H](c2ccc(Cl)cc2Cl)[C@H]([N+](=O)[O-])C1"
    ]
    rxn_types = ["UNK", "UNK"]

    assert len(product_smiles) == len(rxn_types)
    results = proposer.propose(product_smiles, rxn_types, topk=5, beam_size=5)
    print(results)
    """
    List of n[List of topk[List of reactants, score]]
    """


if __name__ == "__main__":
    test()
