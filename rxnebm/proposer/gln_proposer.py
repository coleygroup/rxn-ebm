from gln.test.model_inference import RetroGLN
from proposer_base import Proposer
from typing import List


class GLNProposer(Proposer):
    """GLN proposer, wrapping around GLN.gln.test.model_inference.RetroGLN"""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model = self.build_model(model_path)

    @staticmethod
    def build_model(model_path: str):
        model = RetroGLN(dropbox="./rxnebm/proposer/GLN/dropbox",
                         model_dump=model_path)

        return model

    def propose(self, input_smiles: List[str],
                rxn_types: List[str],
                topk: int = 1, **kwargs) -> List[List]:

        results = []
        for smi, rxn_type in zip(input_smiles, rxn_types):
            result = self.model.run(raw_prod=smi,
                                    beam_size=50,
                                    topk=topk,
                                    rxn_type=rxn_type)
            results.append(result)

        return results
