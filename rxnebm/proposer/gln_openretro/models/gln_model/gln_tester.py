import logging
import numpy as np
import os
import random
import torch
from gln.test.main_test import eval_model
from gln.test.model_inference import RetroGLN
from typing import Dict, List


class GLNTester:
    """Class for GLN Testing"""

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
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.test_output_path = test_output_path

        random.seed(model_args.seed)
        np.random.seed(model_args.seed)
        torch.manual_seed(model_args.seed)

        self.dropbox = processed_data_path

    def test(self):
        """Core of test_all.sh and test_single.sh, adapted from test/main_test.py"""
        if self.model_args.test_all_ckpts:
            logging.info(f"test_all_ckpts flag set to True, testing all checkpoints")
            i = 0
            while True:
                model_dump = os.path.join(self.model_path, f"model-{i}.dump")
                if not os.path.isdir(model_dump):
                    logging.info(f"No checkpoints found at {model_dump}, exiting")
                    break

                logging.info(f"Checkpoints found at {model_dump}, building wrapper")
                self.model_args.model_for_test = model_dump
                model = RetroGLN(self.dropbox, self.model_args.model_for_test)

                logging.info(f"Testing {model_dump}")
                for phase, fname in [("val", self.val_file),
                                     ("test", self.test_file)]:
                    fname_pred = os.path.join(self.test_output_path, f"{phase}-{i}.pred")
                    eval_model(phase, fname, model, fname_pred)
                i += 1
        else:
            model = RetroGLN(self.dropbox, self.model_args.model_for_test)

            logging.info(f"Testing {self.model_args.model_for_test}")
            for phase, fname in [("val", self.val_file),
                                 ("test", self.test_file)]:
                fname_pred = os.path.join(self.test_output_path, f"{phase}.pred")
                eval_model(phase, fname, model, fname_pred)

        # adapted from test/report_test_stats.py
        files = os.listdir(self.test_output_path)

        best_val = 0.0
        best_test = None
        for fname in files:
            if "val-" in fname and "summary" in fname:
                f_test = os.path.join(self.test_output_path, f"test{fname[3:]}")
                if not os.path.isfile(f_test):
                    continue
                with open(os.path.join(self.test_output_path, fname), "r") as f:
                    f.readline()
                    top1 = float(f.readline().strip().split()[-1].strip())
                    if top1 > best_val:
                        best_val = top1
                        best_test = f_test
        assert best_test is not None
        with open(best_test, "r") as f:
            for row in f:
                logging.info(row.strip())
        logging.info(f"best test results: {best_test}")
