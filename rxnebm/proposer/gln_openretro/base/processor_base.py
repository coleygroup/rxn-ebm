import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List


class Processor(ABC):
    """Base class for processor"""

    @abstractmethod
    def __init__(self,
                 model_name: str,
                 model_args,                                        # let's enforce everything to be passed in args
                 model_config: Dict[str, any],                      # or config
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str):
        self.model_name = model_name
        self.model_args = model_args
        self.model_config = model_config
        self.data_name = data_name
        self.raw_data_files = raw_data_files
        self.processed_data_path = processed_data_path

        os.makedirs(self.processed_data_path, exist_ok=True)

    @abstractmethod
    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for all"""
        logging.info("Checking data format before preprocessing")

        for fn in self.raw_data_files:
            if not fn:
                continue
            assert os.path.exists(fn), f"{fn} does not exist!"

    @abstractmethod
    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        pass
