import logging
import numpy as np
import os
import pickle as cp
import random
import torch
import torch.optim as optim
from base.trainer_base import Trainer
from gln.data_process.data_info import load_bin_feats, DataInfo
from gln.graph_logic.logic_net import GraphPath
from gln.training.data_gen import data_gen, worker_softmax
from tqdm import tqdm
from typing import Dict, List


class GLNTrainer(Trainer):
    """Class for GLN Training"""

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
        os.environ["PYTHONHASHSEED"] = str(self.model_args.seed) # do we need this? min htoo added on 23/04/2021
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Override paths and args (temporarily) to be consistent with hardcoding in gln backend
        self.dropbox = processed_data_path
        self.cooked_folder = os.path.join(processed_data_path, f"cooked_{data_name}")
        self.tpl_folder = os.path.join(self.cooked_folder, "tpl-default")
        self.train_file = raw_data_files[0]

        logging.info("Overwriting model args, based on original gln training script")
        self.overwrite_model_args()
        logging.info(f"Updated model args: {self.model_args}")

        DataInfo.init(self.dropbox, self.model_args)
        load_bin_feats(self.dropbox, self.model_args)

    def overwrite_model_args(self):
        """Overwrite model args, adapted from run_mf.sh"""
        # Paths
        self.model_args.dropbox = self.dropbox
        self.model_args.data_name = self.data_name
        self.model_args.tpl_name = "default"
        self.model_args.train_file = self.train_file
        self.model_args.f_atoms = os.path.join(self.dropbox, f"cooked_{self.data_name}", "atom_list.txt")
        # ENV variables
        self.model_args.gm = "mean_field"
        self.model_args.act_func = "relu"
        self.model_args.latent_dim = 128
        self.model_args.embed_dim = 256
        self.model_args.neg_num = 64
        self.model_args.max_lv = 3
        self.model_args.tpl_enc = "deepset"
        self.model_args.subg_enc = "mean_field"
        self.model_args.readout_agg_type = "max"
        self.model_args.retro_during_train = True
        self.model_args.bn = True
        self.model_args.gen_method = "weighted"
        self.model_args.gnn_out = "last"
        self.model_args.neg_sample = "all"
        self.model_args.att_type = "bilinear"
        # Hardcoded at runtime
        self.model_args.fp_degree = 2
        self.model_args.act_last = True
        self.model_args.iters_per_val = 3000
        self.model_args.gpu = 0
        self.model_args.num_cores = len(os.sched_getaffinity(0))
        self.model_args.topk = 50
        self.model_args.beam_size = 50
        self.model_args.num_parts = 1
        # Suggested by gln README
        self.model_args.num_epochs = 10

    def build_train_model(self):
        self.model = GraphPath(self.model_args).to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

    def train(self):
        """Core of run_mf.sh, adapted from training/main.py"""
        logging.info("Creating data generator")
        train_sample_gen = data_gen(self.model_args.num_data_proc,
                                    worker_softmax,
                                    [self.model_args],
                                    max_gen=-1)

        if self.model_args.init_model_dump is not None:
            logging.info(f"Loading checkpoint from {self.model_args.init_model_dump}")
            self.model.load_state_dict(torch.load(self.model_args.init_model_dump))
        else:
            logging.info(f"No checkpoint found, training from scratch")

        logging.info("Creating optimizer")
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_args.learning_rate)

        for epoch in range(self.model_args.num_epochs):
            pbar = tqdm(range(1, 1 + self.model_args.iters_per_val))
            for it in pbar:
                samples = [next(train_sample_gen) for _ in range(self.model_args.batch_size)]
                optimizer.zero_grad()
                loss = self.model(samples)
                loss.backward()

                if self.model_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.model_args.grad_clip)
                optimizer.step()
                pbar.set_description(f"epoch {epoch + it / self.model_args.iters_per_val: .2f}, "
                                     f"loss {loss.item(): .4f}")

            if epoch % self.model_args.epochs2save == 0:
                out_folder = os.path.join(self.model_path, f"model-{epoch}.dump")
                os.makedirs(out_folder, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(out_folder, "model.dump"))
                with open(os.path.join(out_folder, "args.pkl"), "wb") as f:
                    cp.dump(self.model_args, f, cp.HIGHEST_PROTOCOL)

    def test(self):
        """GLN has an additional wrapper on top of the train model for testing. See gln_tester.py"""
        raise NotImplementedError
