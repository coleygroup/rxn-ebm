from typing import Dict, List

from rxnebm.proposer.proposer_base import Proposer
from rxnebm.proposer.retrosim_model import Retrosim


class RetrosimProposer(Proposer):
    """ Retrosim proposer """

    def __init__(self, retrosim_config: Dict) -> None:
        super().__init__() 
        self.model = self.build_model(retrosim_config)

    @staticmethod
    def build_model(retrosim_config: Dict) -> Retrosim:
        model = Retrosim(**retrosim_config)
        return model

    def propose(self, 
                input_smiles: List[str],
                topk: int = 1, 
                max_prec: int = 200, 
                # rxn_types: List[str] = None, 
                **kwargs) -> List[Dict[str, List]]:
        # TODO: add support for proposal when rxn_type is known (need to modify Retrosim.py)

        results = []
        # for smi, rxn_type in zip(input_smiles, rxn_types):
        for smi in input_smiles:
            result_dict = {} 
            result = self.model.propose_one_helper(
                                    prod_smiles=smi,  
                                    results=result_dict,
                                    topk=topk, 
                                    max_prec=max_prec
                                    # rxn_type=rxn_type
                                    )
            results.append(result)

        # TODO: consider merging List[Dict] into just one Dict 
        return results
