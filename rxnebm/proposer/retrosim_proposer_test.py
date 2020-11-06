from rxnebm.proposer.retrosim_config import retrosim_config
from rxnebm.proposer.retrosim_proposer import RetrosimProposer

def test():
    proposer = RetrosimProposer(retrosim_config)
    product_smiles = [
        "[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]",
        "[Br:1][c:12]1[cH:11][cH:10][c:9]([OH:15])[c:8]([Cl:7])[c:13]1[Cl:14]"
    ] 
 
    results = proposer.propose(product_smiles, topk=5, max_prec=200)
    print(results)
    """
    List of n[{"reactants": List of topk reactants}]
    """

if __name__ == "__main__":
    test()