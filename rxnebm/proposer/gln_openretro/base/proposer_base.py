import os
from abc import ABC, abstractmethod
from typing import Dict, List


class Proposer(ABC):
    """Base class for proposer"""

    @abstractmethod
    def __init__(self,
                 model_name: str,
                 model_args,                                        # let's enforce everything to be passed in args
                 model_config: Dict[str, any],                      # or config
                 data_name: str,
                 processed_data_path: str,
                 model_path: str):
        self.model_name = model_name
        self.model_args = model_args
        self.model_config = model_config
        self.data_name = data_name
        self.processed_data_path = processed_data_path
        self.model_path = model_path

        assert os.path.exists(model_path), f"{model_path} does not exist!"

    @abstractmethod
    def build_predict_model(self):
        pass

    @abstractmethod
    def propose(self, input_smiles: List[str],
                rxn_types: List[str],
                topk: int = 1,
                **kwargs) -> List[Dict[str, List]]:
        pass

    # propose_batch() is optional
