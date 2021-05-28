import logging
import numpy as np
import os
import random
import torch
from base.trainer_base import Trainer
from onmt.bin.train import train as onmt_train
from typing import Dict, List


class TransformerTrainer(Trainer):
    """Class for Transformer Training"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 model_path: str):
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         processed_data_path=processed_data_path,
                         model_path=model_path)

        random.seed(self.model_args.seed)
        np.random.seed(self.model_args.seed)
        torch.manual_seed(self.model_args.seed)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Overwriting model args, (hardcoding essentially)")
        self.overwrite_model_args()
        logging.info(f"Updated model args: {self.model_args}")

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        self.model_args.data = os.path.join(self.processed_data_path, "bin")
        self.model_args.save_model = os.path.join(self.model_path, "model")

    def build_train_model(self):
        logging.info("For onmt training, models are built implicitly.")

    def train(self):
        """A wrapper to onmt.bin.train()"""
        onmt_train(self.model_args)

    def test(self):
        """TBD"""
        raise NotImplementedError
