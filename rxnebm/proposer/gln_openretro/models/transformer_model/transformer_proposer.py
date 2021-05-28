import torch
from base.proposer_base import Proposer
from models.transformer_model.transformer_processor import smi_tokenizer
from onmt.translate.translation_server import ServerModel as ONMTServerModel
from typing import Dict, List


class TransformerProposer(Proposer):
    """Transformer proposer, wrapping around gln.test.model_inference.RetroGLN"""

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
        self.n_best = self.model_config["n_best"]
        self.beam_size = self.model_config["beam_size"]

        self.build_predict_model()

    def build_predict_model(self):
        # Clean up model_config to pass translate_opt check
        self.model_config["models"] = self.model_path
        del self.model_config["model_name"]
        del self.model_config["processed_data_path"]
        del self.model_config["model_path"]

        self.model = ONMTServerModel(
            opt=self.model_config,
            model_id=0,
            load=True
        )

    def propose(self, input_smiles: List[str],
                **kwargs) -> List[Dict[str, List]]:

        inputs = [{"src": smi_tokenizer(smi)} for smi in input_smiles]

        reactants, scores, _, _, _ = self.model.run(inputs=inputs)

        results = []
        for i, prod in enumerate(input_smiles):             # essentially reshaping (b*n_best,) into (b, n_best)
            start = self.n_best * i
            end = self.n_best * (i + 1)
            result = {
                "reactants": ["".join(r.split()) for r in reactants[start:end]],
                "scores": scores[start:end]
            }
            results.append(result)

        return results
