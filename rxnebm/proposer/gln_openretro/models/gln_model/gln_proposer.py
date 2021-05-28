import torch
from base.proposer_base import Proposer
from gln.test.model_inference import RetroGLN
from typing import Dict, List


class GLNProposer(Proposer):
    """GLN proposer, wrapping around gln.test.model_inference.RetroGLN"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 processed_data_path: str,
                 model_path: str) -> None:
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         processed_data_path=processed_data_path,
                         model_path=model_path)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropbox = processed_data_path

        self.build_predict_model()

    def build_predict_model(self):
        self.model = RetroGLN(dropbox=self.dropbox,
                              model_dump=self.model_path)

    def propose(self, input_smiles: List[str],
                rxn_types: List[str],
                topk: int = 1,
                beam_size: int = 50,
                **kwargs) -> List[Dict[str, List]]:

        results = []
        for smi, rxn_type in zip(input_smiles, rxn_types):
            result = self.model.run(raw_prod=smi,
                                    beam_size=beam_size,
                                    topk=topk,
                                    rxn_type=rxn_type)
            results.append(result)

        return results
