import logging
import numpy as np
import os
import random
import torch
from onmt.bin.translate import translate as onmt_translate
from rdkit import Chem
from typing import Dict, List


def canonicalize_smiles(smiles: str):
    """Adapted from Molecular Transformer"""
    smiles = "".join(smiles.split())

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ""


class TransformerTester:
    """Class for Transformer Testing"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 model_path: str,
                 test_output_path: str):

        self.model_name = model_name
        self.model_args = model_args
        self.model_config = model_config
        self.data_name = data_name
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.test_output_path = test_output_path

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
        self.model_args.models = [self.model_path]
        self.model_args.src = os.path.join(self.processed_data_path, "src-test.txt")
        self.model_args.output = os.path.join(self.test_output_path, "predictions_on_test.txt")

    def test(self):
        """Actual file-based testing, a wrapper to onmt.bin.translate()"""
        onmt_translate(self.model_args)
        self.score_predictions()

    def score_predictions(self):
        """Adapted from Molecular Transformer"""
        logging.info("Done generation, scoring predictions")
        with open(os.path.join(self.processed_data_path, "tgt-test.txt"), "r") as f:
            gts = f.readlines()
        with open(self.model_args.output, "r") as f:
            predictions = f.readlines()

        n_best = self.model_args.n_best
        assert len(gts) == (len(predictions) / n_best), \
            f"File length mismatch! Ground truth total: {len(gts)}, " \
            f"prediction total: {len(predictions)}, n_best: {n_best}"

        accuracies = np.zeros([len(gts), n_best], dtype=np.float32)

        for i, gt in enumerate(gts):
            gt_smiles = [canonicalize_smiles(smi) for smi in gt.strip().split(".")]

            for j in range(n_best):
                prediction = predictions[i*n_best+j]
                predicted_smiles = [canonicalize_smiles(smi) for smi in prediction.strip().split(".")]

                if len(gt_smiles) == len(predicted_smiles) and all(smi in gt_smiles for smi in predicted_smiles):
                    accuracies[i, j:] = 1.0             # accuracies on or after jth rank will be 1.0
                    break

        # Log statistics
        mean_accuracies = np.mean(accuracies, axis=0)
        for n in range(n_best):
            logging.info(f"Top {n+1} accuracy: {mean_accuracies[n]}")
