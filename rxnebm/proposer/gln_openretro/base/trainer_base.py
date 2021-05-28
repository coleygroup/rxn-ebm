import os
from abc import ABC, abstractmethod
from typing import Dict


class Trainer(ABC):
    """Base class for trainer"""

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

        assert os.path.exists(processed_data_path), f"{processed_data_path} does not exist!"

        os.makedirs(model_path, exist_ok=True)

    @abstractmethod
    def build_train_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    # test() is optional
