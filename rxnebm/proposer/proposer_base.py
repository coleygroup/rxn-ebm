from abc import ABC, abstractmethod
from typing import List


class Proposer(ABC):
    """Base model for proposer"""

    @abstractmethod
    def build_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def propose(self, input_smiles: List[str],
                rxn_types: List[str],
                topk: int = 1, **kwargs) -> List[List]:
        pass
