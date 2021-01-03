import logging
import traceback
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 

from rxnebm.data import dataset, dataset_utils
from rxnebm.experiment import expt_utils
from rxnebm.model import model_utils

Tensor = torch.Tensor

class Experiment:
    """
    TODO: representation: ['graph']
    TODO: wandb/tensorboard for hyperparameter tuning

    Parameters
    ----------
    representation : str ['fingerprint']
        which representation to use. affects which version of AugmentedData & ReactionDataset gets called.
        if not using 'fingerprint', cannot use Bit Augmentor. 

    load_checkpoint : Optional[bool] (Default = False)
        whether to load from a previous checkpoint.
        if True: load_optimizer, load_stats & begin_epoch must be provided
    """

    def __init__(
        self,
        args,
        model: nn.Module,
        model_name: str,
        model_args: dict,
        augmentations: dict,
        onthefly: Optional[bool] = False,
        debug: Optional[bool] = True, 
        device: Optional[str] = None,
        distributed: Optional[bool] = False,
        root: Optional[Union[str, bytes, os.PathLike]] = None,
        load_checkpoint: Optional[bool] = False,
        saved_optimizer: Optional[torch.optim.Optimizer] = None,
        saved_stats: Optional[dict] = None,
        begin_epoch: Optional[int] = None,
        vocab: Dict[str, int] = None,
        **kwargs
    ):
        self.args = args
        self.debug = debug
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.best_epoch = 0  # will be automatically assigned after 1 epoch
        self.vocab = vocab

        self.early_stop = args.early_stop
        if self.early_stop:
            self.early_stop_criteria = args.early_stop_criteria
            self.early_stop_min_delta = args.early_stop_min_delta
            self.early_stop_patience = args.early_stop_patience
            self.min_val_loss = float("+inf")  
            self.max_val_acc = float("-inf") 
            self.wait = 0  # counter for _check_earlystop()
        else:
            self.early_stop_criteria = None
            self.early_stop_min_delta = None
            self.early_stop_patience = None
            self.min_val_loss = None
            self.max_val_acc = None
            self.wait = None

        self.num_workers = args.num_workers
        self.checkpoint = args.checkpoint
        self.random_seed = args.random_seed

        if root:
            self.root = Path(root)
        else:
            self.root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        self.checkpoint_folder = Path(self.args.checkpoint_folder)

        self.expt_name = args.expt_name
        self.augmentations = augmentations
        self.representation = args.representation
        if self.representation != 'fingerprint' and 'bit' in augmentations:
            raise RuntimeError('Bit Augmentor is only compatible with fingerprint representation!')
        logging.info(f"\nInitialising experiment: {self.expt_name}")
        logging.info(f"Augmentations: {self.augmentations}")

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.model = model
        self.model_name = model_name # self.model.model_repr
        self.model_args = model_args
        self.distributed = distributed  # affects how checkpoint is saved & how weights should be loaded

        if saved_optimizer is not None:
            self.optimizer_name = str(self.args.optimizer).split(' ')[0]
        else:
            self.optimizer_name = args.optimizer
        self.lr_scheduler_name = args.lr_scheduler

        self._collate_args()

        self.rxn_smis_filenames = defaultdict(str)
        if self.args.do_pretrain:
            for phase in ['train', 'valid', 'test']:
                self.rxn_smis_filenames[phase] = f"{self.args.rxn_smis_file_prefix}_{phase}.pickle"

        self.proposals_data = {}
        self.proposals_csv_filenames = defaultdict(str)
        if self.args.do_finetune:
            for phase in ['train', 'valid', 'test']:
                self.proposals_csv_filenames[phase] = self.args.proposals_csv_file_prefix + f"_{phase}.csv"
                if phase != 'train': # this is just for visualization purpose during val/test
                    self.proposals_data[phase] = pd.read_csv(self.root / self.proposals_csv_filenames[phase], index_col=None, dtype='str')
                    self.proposals_data[phase] = self.proposals_data[phase].drop(['orig_rxn_smi'], axis=1).values

        # take into account original probs/scores from retro models
        self.prob_filenames = defaultdict(str)
        if self.args.prob_file_prefix: # will be loaded in Dataset file
            for phase in ['train', 'valid', 'test']:
                self.prob_filenames[phase] = self.args.prob_file_prefix + f"_{phase}.npy"

        if self.representation == 'fingerprint':
            self._init_fp_dataloaders(
                precomp_file_prefix=args.precomp_file_prefix,
                onthefly=onthefly,
                smi_to_fp_dict_filename=args.smi_to_fp_dict_filename,
                fp_to_smi_dict_filename=args.fp_to_smi_dict_filename,
                mol_fps_filename=args.mol_fps_filename,
                search_index_filename=args.search_index_filename,
            )
            self.maxk = self.train_loader.dataset.data.shape[-1] // self.train_loader.dataset.input_dim
        elif self.representation == "smiles":
            self._init_smi_dataloaders(onthefly=onthefly)
            self.maxk = len(self.train_loader.dataset._rxn_smiles_with_negatives[0])
        else:
            raise NotImplementedError('Please add self.maxk for the current representation; '
                                      'we need maxk to calculate topk_accs up to the maximum allowed value '
                                      '(otherwise torch will raise index out of range)')
            # e.g. we can't calculate top-50 acc if we only have 1 positive + 10 negatives per minibatch

        if load_checkpoint:
            self._load_checkpoint(saved_optimizer, saved_stats, begin_epoch)
        else:
            self._init_optimizer_and_stats()

        self.train_losses = []
        self.val_losses = []
        self.train_topk_accs = {}  
        self.val_topk_accs = {}  
        self.test_topk_accs = {} 
        self.k_to_calc = [1, 5, 10, 50]     # [1, 2, 3, 5, 10, 20, 50, 100] seems to slow down training...?
        k_not_to_calc = []
        for k in self.k_to_calc:
            if k > self.maxk:
                k_not_to_calc.append(k)
            else:   # init empty lists; 1 value will be appended to each list per epoch
                self.train_topk_accs[k] = []
                self.val_topk_accs[k] = []  
                self.test_topk_accs[k] = None    
        for k in k_not_to_calc:
            self.k_to_calc.remove(k)
        
        self.energies = {} # for self.get_topk_acc
        self.true_ranks = {} # for self.get_topk_acc (finetuning)

        model_utils.seed_everything(args.random_seed)

    def __repr__(self):
        return f"Experiment name: {self.expt_name}, \
                with augmentations: {self.augmentations}" 

    def _collate_args(self): 
        if self.representation == 'fingerprint':
            self.fp_args = {
                "rxn_type": self.args.rxn_type,
                "fp_type": self.args.fp_type,
                "rctfp_size": self.args.rctfp_size,
                "prodfp_size": self.args.prodfp_size,
                "difffp_size": self.args.difffp_size,
                "fp_radius": self.args.fp_radius,
            }
        else:
            self.fp_args = None
        self.train_args = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler_name,
            "lr_scheduler_criteria": self.args.lr_scheduler_criteria,
            "lr_scheduler_factor": self.args.lr_scheduler_factor,
            "lr_scheduler_patience": self.args.lr_scheduler_patience,
            "early_stop": self.early_stop,
            "early_stop_criteria": self.early_stop_criteria,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "num_workers": self.num_workers,
            "checkpoint": self.checkpoint,
            "random_seed": self.random_seed,
            "expt_name": self.expt_name,
            "device": self.device,
            "model_name": self.model_name,
            "distributed": self.distributed, 
            "optimizer": self.optimizer_name, 
        }

    def _load_checkpoint(
        self,
        saved_optimizer: torch.optim.Optimizer,
        saved_stats: dict,
        begin_epoch: int,
    ):
        logging.info("Loading checkpoint...")

        if saved_optimizer is None:
            raise ValueError("load_checkpoint requires saved_optimizer!")
        self.optimizer = saved_optimizer  # load optimizer w/ state dict from checkpoint
        if self.lr_scheduler_name == 'ReduceLROnPlateau':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            if self.args.lr_scheduler_criteria == 'acc':
                mode = 'max'
            elif self.args.lr_scheduler_criteria == 'loss':
                mode = 'min'
            else:
                raise ValueError(f"Unsupported lr_scheduler_criteria {self.args.lr_scheduler_criteria}")
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer, 
                                        mode=mode, 
                                        factor=self.args.lr_scheduler_factor, 
                                        patience=self.args.lr_scheduler_patience, 
                                        verbose=True
                                        ) 
        elif self.lr_scheduler_name == 'CosineAnnealingWarmRestarts':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer, 
                                        T_0=self.args.lr_scheduler_T_0
                                        )
        else:
            self.lr_scheduler = None 
        # NOTE/TODO: lr_scheduler.load_state_dict(checkpoint['scheduler'])
        # https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/torchtools.py#L104
        # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208/2

        if saved_stats is None:
            raise ValueError("load_checkpoint requires saved_stats!")
        self.stats_filename = (
            self.checkpoint_folder / f"{self.model_name}_{self.expt_name}_stats.pkl"
        )
        self.stats = {
            "model_args": self.model_args, # from saved_stats, handled in trainEBM.py
            "fp_args": self.fp_args, # from saved_stats, handled in trainEBM.py
            "train_args": self.train_args, # new, provided as args
            "augmentations": self.augmentations, # can be new or same, provided as args
            "train_time": 0, # reset to 0
        }
        if begin_epoch is None:
            raise ValueError("load_checkpoint requires begin_epoch!")
        self.begin_epoch = begin_epoch

    def _init_optimizer_and_stats(self): 
        logging.info("Initialising optimizer & stats...")
        self.optimizer = model_utils.get_optimizer(self.optimizer_name)(self.model.parameters(), lr=self.learning_rate) 
        if self.lr_scheduler_name == 'ReduceLROnPlateau':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            if self.args.lr_scheduler_criteria == 'acc':
                mode = 'max'
            elif self.args.lr_scheduler_criteria == 'loss':
                mode = 'min'
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer, 
                                        mode=mode, 
                                        factor=self.args.lr_scheduler_factor, 
                                        patience=self.args.lr_scheduler_patience, 
                                        verbose=True
                                        ) 
        elif self.lr_scheduler_name == 'CosineAnnealingWarmRestarts':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer, 
                                        T_0=self.args.lr_scheduler_T_0
                                        )
        else:
            logging.info('Not using any LR Scheduler!')
            self.lr_scheduler = None 

        self.begin_epoch = 0 
        self.stats = {
            "model_args": self.model_args,
            "fp_args": self.fp_args,
            "train_args": self.train_args,
            "augmentations": self.augmentations,
            "train_time": 0,
        }
        self.stats_filename = (
            self.checkpoint_folder / f"{self.model_name}_{self.expt_name}_stats.pkl"
        )

    def _init_fp_dataloaders(
        self,
        precomp_file_prefix: str,
        onthefly: bool,
        smi_to_fp_dict_filename: str,
        fp_to_smi_dict_filename: str,
        mol_fps_filename: str,
        search_index_filename: str,
    ):
        logging.info("Initialising dataloaders...")
        if "cos" in self.augmentations:
            worker_init_fn = expt_utils._worker_init_fn_nmslib
        else:
            worker_init_fn = expt_utils._worker_init_fn_default

        augmented_data = None
        precomp_rxnfp_filenames = defaultdict(str)
        if onthefly and args.do_pretrain:
                augmented_data = dataset.AugmentedDataFingerprints(
                    self.augmentations,
                    smi_to_fp_dict_filename,
                    fp_to_smi_dict_filename,
                    mol_fps_filename,
                    search_index_filename,
                )
        else: # precomputed rxn fp files (augmented data / retro proposals)
            for phase in ["train", "valid", "test"]:
                precomp_rxnfp_filenames[phase] = precomp_file_prefix + f"_{phase}.npz"

        if self.args.rxn_type == "sep":
            input_dim = self.args.rctfp_size + self.args.prodfp_size
        elif self.args.rxn_type == "diff":
            input_dim = self.args.rctfp_size
        elif self.args.rxn_type == 'hybrid':
            input_dim = self.args.prodfp_size + self.args.difffp_size
        elif self.args.rxn_type == 'hybrid_all':
            input_dim = self.args.rctfp_size + self.args.prodfp_size + self.args.difffp_size

        train_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_rxnfp_filenames["train"],
            proposals_csv_filename=self.proposals_csv_filenames["train"],
            rxn_smis_filename=self.rxn_smis_filenames["train"],
            onthefly=onthefly,
            augmented_data=augmented_data,
            prob_filename=self.prob_filenames['train']
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=True,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last
        )
        self.train_size = len(train_dataset)

        val_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_rxnfp_filenames["valid"],
            proposals_csv_filename=self.proposals_csv_filenames["valid"],
            rxn_smis_filename=self.rxn_smis_filenames["valid"],
            onthefly=onthefly,
            augmented_data=augmented_data,
            prob_filename=self.prob_filenames['valid']
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last
        )
        self.val_size = len(val_dataset)

        test_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_rxnfp_filenames["test"],
            proposals_csv_filename=self.proposals_csv_filenames["test"],
            rxn_smis_filename=self.rxn_smis_filenames["test"],
            onthefly=onthefly,
            augmented_data=augmented_data,
            prob_filename=self.prob_filenames['test']
        )
        self.test_loader = DataLoader(
            test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last
        )
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset

    def _get_smi_dl(self, phase: str, shuffle: bool = False):
        rxn_smis_filename, proposals_csv_filename = None, None
        
        _data = dataset.ReactionDatasetSMILES(
            self.args,
            phase=phase,
            augmentations=self.augmentations,
            # precomp_rxnsmi_filename=None, # TODO? add later if needed
            rxn_smis_filename=self.rxn_smis_filenames[phase],
            proposals_csv_filename=self.proposals_csv_filenames[phase],
            onthefly=True,
            prob_filename=self.prob_filenames[phase]
        )

        if self.args.do_compute_graph_feat:
            collate_fn = dataset_utils.graph_collate_fn_builder(
                self.device, debug=False)
        else:
            collate_fn = dataset_utils.seq_collate_fn_builder(
                self.device, self.vocab, self.args.max_seq_len, debug=False)

        _loader = DataLoader(
            _data,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_fn,
            drop_last=self.args.drop_last
        )

        _size = len(_data)
        del _data

        return _loader, _size

    def _init_smi_dataloaders(self, onthefly: bool):
        logging.info("Initialising SMILES dataloaders...")
        if onthefly:
            self.train_loader, self.train_size = self._get_smi_dl(phase="train", shuffle=True)
            self.val_loader, self.val_size = self._get_smi_dl(phase="valid", shuffle=False)
            self.test_loader, self.test_size = self._get_smi_dl(phase="test", shuffle=False)
        else:
            raise NotImplementedError('Only onthefly available for smi loader')

    def _check_earlystop(self, current_epoch):
        if self.early_stop_criteria == 'loss': 
            if self.min_val_loss - self.val_losses[-1] < self.early_stop_min_delta:
                if self.early_stop_patience <= self.wait:
                    logging.info(
                        f"\nEarly stopped at the end of epoch: {current_epoch}, \
                    train loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_topk_accs[1][-1]:.4f}, \
                    \nval loss: {self.val_losses[-1]:.4f}, top-1 val acc: {self.val_topk_accs[1][-1]:.4f}"
                    )
                    self.stats["early_stop_epoch"] = current_epoch
                    self.to_break = 1  # will break loop
                else:
                    self.wait += 1
                    logging.info(f"\nDecrease in val loss < early stop min delta {self.early_stop_min_delta}, \
                                patience count: {self.wait}")
            else:
                self.wait = 0
                self.min_val_loss = min(self.min_val_loss, self.val_losses[-1])

        elif self.early_stop_criteria.split('_')[-1] == 'acc':
            k = int(self.early_stop_criteria.split('_')[0][-1:])
            val_acc_to_compare = self.val_topk_accs[k][-1] 

            if self.max_val_acc - val_acc_to_compare > self.early_stop_min_delta:
                if self.early_stop_patience <= self.wait:
                    logging.info(
                        f"\nEarly stopped at the end of epoch: {current_epoch}, \
                    \ntrain loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_topk_accs[1][-1]:.4f}, \
                    \nval loss: {self.val_losses[-1]:.4f}, top-1 val acc: {self.val_topk_accs[1][-1]:.4f} \
                    \n"
                    )
                    self.stats["early_stop_epoch"] = current_epoch
                    self.to_break = 1  # will break loop
                else:
                    self.wait += 1
                    logging.info(
                        f'\nIncrease in top-{k} val acc < early stop min delta {self.early_stop_min_delta}, \
                        \npatience count: {self.wait} \
                        \n')
            else:
                self.wait = 0
                self.max_val_acc = max(self.max_val_acc, val_acc_to_compare)

    def _update_stats(self):
        self.stats["train_time"] = (
            self.stats["train_time"] + (time.time() - self.start) / 60
        )  # in minutes
        self.start = time.time()
        # a list with one value for each epoch
        self.stats["train_losses"] = self.train_losses 
        self.stats["train_topk_accs"] = self.train_topk_accs 

        self.stats["val_losses"] = self.val_losses 
        self.stats["val_topk_accs"] = self.val_topk_accs 

        self.stats["min_val_loss"] = self.min_val_loss
        self.stats["max_val_acc"] = self.max_val_acc
        self.stats["best_epoch"] = self.best_epoch
        torch.save(self.stats, self.stats_filename)

    def _checkpoint_model_and_opt(self, current_epoch: int):
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        checkpoint_dict = {
            "epoch": current_epoch,  # epochs are 0-indexed
            "model_name": self.model_name,
            "state_dict": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "stats": self.stats,
        }
        checkpoint_filename = (
            self.checkpoint_folder
            / f"{self.model_name}_{self.expt_name}_checkpoint_{current_epoch:04d}.pth.tar"
        )
        torch.save(checkpoint_dict, checkpoint_filename)

    def _one_batch(self, batch: Tensor, mask: Tensor, probs: Optional[Tensor] = None, backprop: bool = True):
        """
        Passes one batch of samples through model to get energies & loss
        Does backprop if training 
        """
        # for p in self.model.parameters():
        #     p.grad = None  # faster, equivalent to self.model.zero_grad()
        self.model.zero_grad()
        energies = self.model(batch, probs)  # size N x K
        # print('\n', '#'*10, 'energies (before masking)', '#'*10)
        # print(energies.shape)
        # print(energies)

        # replace all-zero vectors with float('inf'), making those gradients 0 on backprop
        energies = torch.where(mask, energies, torch.tensor([float('inf')], device=mask.device))
        # print('\n', '#'*10, 'mask', '#'*10)
        # print(mask.shape)
        # print(mask)

        # print('\n', '#'*10, 'energies (after masking)', '#'*10)
        # print(energies.shape)
        # print(energies)
        if backprop:
            # for training only: positives are the 0-th index of each minibatch (row)
            loss = (energies[:, 0] + torch.logsumexp(-energies, dim=1)).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item(), energies.cpu().detach() 
        else:
            # if fine-tuning: validation/testing, cannot calculate loss naively
            # need the true_ranks of precursors from retrosynthesis models
            return energies.cpu().detach() 
 
    def train(self):
        self.start = time.time()  # timeit.default_timer()
        self.to_break = 0
        for epoch in range(self.begin_epoch, self.epochs + self.begin_epoch):
            self.model.train()
            train_loss, train_correct_preds = 0, defaultdict(int)
            epoch_train_size = 0
            train_loader = tqdm(self.train_loader, desc='training...')
            for i, batch in enumerate(train_loader):
                # if i > 4:
                #     break
                batch_data = batch[0]
                # print(f'Check batch_data type: {isinstance(batch_data, tuple)}') # returns True for TransformerEBM
                if not isinstance(batch_data, tuple):
                    batch_data = batch_data.to(self.device)
                if self.model_name == 'TransformerEBM':
                    batch_data = (batch_data, 'train')
                batch_mask = batch[1].to(self.device)
                if self.args.prob_file_prefix:
                    batch_probs = batch[3].to(self.device)
                    batch_loss, batch_energies = self._one_batch(
                        batch_data, batch_mask, batch_probs, backprop=True
                    )
                else:
                    batch_loss, batch_energies = self._one_batch(
                        batch_data, batch_mask, backprop=True
                    )
                train_loss += batch_loss
                train_batch_size = batch_energies.shape[0]
                epoch_train_size += train_batch_size

                if self.lr_scheduler_name == 'CosineAnnealingWarmRestarts':
                    self.lr_scheduler.step(epoch + i / self.train_size - self.args.lr_scheduler_epoch_offset) # - self.begin_epoch)

                for k in self.k_to_calc: # various top-k accuracies
                    # index with lowest energy is what the model deems to be the most feasible rxn 
                    batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1] 
                    # for training, true rxn is always at 0th-index 
                    batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0]
                    train_correct_preds[k] += batch_correct_preds
                    if k == 1:
                        batch_top1_acc = batch_correct_preds / train_batch_size
                    elif k == 5:
                        batch_top5_acc = batch_correct_preds / train_batch_size
                    elif k == 10:
                        batch_top10_acc = batch_correct_preds / train_batch_size
                if 5 not in self.k_to_calc:
                    batch_top5_acc = np.nan 
                if 10 not in self.k_to_calc:
                    batch_top10_acc = np.nan 
                train_loader.set_description(f"training...loss={batch_loss/train_batch_size:.4f}, top-1 acc={batch_top1_acc:.4f}, top-5 acc={batch_top5_acc:.4f}, top-10 acc={batch_top10_acc:.4f}")
                train_loader.refresh()
            
            for k in self.k_to_calc:     
                self.train_topk_accs[k].append(train_correct_preds[k] / epoch_train_size) # self.train_size ) 
            self.train_losses.append(train_loss / epoch_train_size) # self.train_size)

            # validation
            self.model.eval()
            with torch.no_grad(): 
                val_loss, val_correct_preds = 0, defaultdict(int)
                val_loader = tqdm(self.val_loader, desc='validating...')
                epoch_val_size = 0
                for i, batch in enumerate(val_loader):
                    # if i > 3:
                    #     break
                    batch_data = batch[0]
                    if not isinstance(batch_data, tuple):
                        batch_data = batch_data.to(self.device)
                    if self.model_name == 'TransformerEBM':
                        batch_data = (batch_data, 'valid')
                    batch_mask = batch[1].to(self.device)
                    if self.args.prob_file_prefix:
                        batch_probs = batch[3].to(self.device)
                        batch_energies = self._one_batch(
                            batch_data, batch_mask, batch_probs, backprop=False
                        )
                    else:
                        batch_energies = self._one_batch(
                            batch_data, batch_mask, backprop=False
                        )
                    val_batch_size = batch_energies.shape[0]
                    epoch_val_size += val_batch_size

                    # for validation/test data, true rxn may not be present!
                    # only provide for finetuning step on retro proposals
                    if self.args.do_finetune and self.debug:
                        batch_idx = batch[2] # List
                        # print('\n', '#'*10, 'batch_idx', '#'*10)
                        # print(len(batch_idx))
                        # print(batch_idx)

                        batch_true_ranks_array = self.proposals_data['valid'][batch_idx, 2].astype('int')
                        # print('\n', '#'*10, 'batch_true_ranks_array', '#'*10)
                        # print(batch_true_ranks_array.shape)
                        # print(batch_true_ranks_array)

                        batch_true_ranks_valid = batch_true_ranks_array[batch_true_ranks_array < self.args.minibatch_eval] # != 9999]
                        # print('\n', '#'*10, 'batch_true_ranks_valid', '#'*10)
                        # print(batch_true_ranks_valid.shape)
                        # print(batch_true_ranks_valid)
                        
                        batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                        # print('\n', '#'*10, 'batch_true_ranks', '#'*10)
                        # print(batch_true_ranks.shape)
                        # print(batch_true_ranks)
                        
                        # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation
                        # (bcos nothing in the numerator, loss is undefined)
                        loss_numerator = batch_energies[
                            np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval], #!= 9999],
                            batch_true_ranks_valid
                        ]
                        # print('\n', '#'*10, 'loss_numerator', '#'*10)
                        # print(loss_numerator.shape)
                        # print(loss_numerator)

                        loss_denominator = batch_energies[
                            np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval], #!= 9999],
                            :
                        ]
                        # print('\n', '#'*10, 'loss_denominator', '#'*10)
                        # print(loss_denominator.shape)
                        # print(loss_denominator)

                        batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item() 

                        for k in self.k_to_calc:
                            # index with lowest energy is what the model deems to be the most feasible rxn
                            batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]
                            # print('\n', '#'*10, 'batch_preds', '#'*10)
                            # print(batch_preds.shape)
                            # print(batch_preds)

                            batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                            val_correct_preds[k] += batch_correct_preds

                            if k == 1:
                                batch_top1_acc = batch_correct_preds / val_batch_size
                                if self.debug: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                                    try:
                                        for j in range(i * self.batch_size, (i+1) * self.batch_size):
                                            if j % (self.val_size // 5) == 0:  # peek at a random sample of current batch to monitor training progress
                                                sample_idx = random.sample(list(range(self.batch_size)), k=1)[0]
                                                sample_true_rank = batch_true_ranks_array[sample_idx]
                                                sample_pred_rank = batch_preds[sample_idx, 0].item() 
                                                sample_true_prod = self.proposals_data['valid'][batch_idx[sample_idx], 0]
                                                sample_true_prec = self.proposals_data['valid'][batch_idx[sample_idx], 1]

                                                sample_cand_precs = self.proposals_data['valid'][batch_idx[sample_idx], 3:]
                                                sample_pred_prec = sample_cand_precs[batch_preds[sample_idx]]
                                                sample_orig_prec = sample_cand_precs[0]
                                                logging.info(f'\ntrue product:            {sample_true_prod}')
                                                logging.info(f'pred precursor (rank {sample_pred_rank}): {sample_pred_prec}')
                                                logging.info(f'true precursor (rank {sample_true_rank}): {sample_true_prec}')
                                                logging.info(f'orig precursor (rank 0): {sample_orig_prec}\n')
                                                break
                                    except Exception as e: # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                                        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                                        logging.info("".join(tb_str))
                                        logging.info('\nIndex out of range (last minibatch)')
                            elif k == 5:
                                batch_top5_acc = batch_correct_preds / val_batch_size
                            elif k == 10:
                                batch_top10_acc = batch_correct_preds / val_batch_size
                    else:       # for pre-training step w/ synthetic data, 0-th index is the positive rxn
                        batch_true_ranks = 0
                        batch_loss = (batch_energies[:, 0] + torch.logsumexp(-batch_energies, dim=1)).sum().item() 

                        # calculate top-k acc assuming true index is 0 (for pre-training step)
                        for k in self.k_to_calc:
                            batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1] 
                            batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0] 
                            val_correct_preds[k] += batch_correct_preds

                            if k == 1:
                                batch_top1_acc = batch_correct_preds / val_batch_size
                            elif k == 5:
                                batch_top5_acc = batch_correct_preds / val_batch_size
                            elif k == 10:
                                batch_top10_acc = batch_correct_preds / val_batch_size
                    if 5 not in self.k_to_calc:
                        batch_top5_acc = np.nan
                    if 10 not in self.k_to_calc:
                        batch_top10_acc = np.nan 
                    val_loader.set_description(f"validating...loss={batch_loss/val_batch_size:.4f}, top-1 acc={batch_top1_acc:.4f}, top-5 acc={batch_top5_acc:.4f}, top-10 acc={batch_top10_acc:.4f}")
                    val_loader.refresh()
                    val_loss += batch_loss  
                
                for k in self.k_to_calc:
                    self.val_topk_accs[k].append(val_correct_preds[k] / epoch_val_size) # self.val_size)
                self.val_losses.append(val_loss / epoch_val_size) # self.val_size)

            # track best_epoch to facilitate loading of best checkpoint
            if self.early_stop:
                if self.early_stop_criteria == 'loss':
                    if self.val_losses[-1] < self.min_val_loss:
                        self.best_epoch = epoch
                elif self.early_stop_criteria.split('_')[-1] == 'acc':
                    k = int(self.early_stop_criteria.split('_')[0][-1:])
                    val_acc_to_compare = self.val_topk_accs[k][-1]
                    if val_acc_to_compare > self.max_val_acc:
                        self.best_epoch = epoch

            self._update_stats()
            if self.checkpoint:
                self._checkpoint_model_and_opt(current_epoch=epoch)
            if self.early_stop:
                self._check_earlystop(current_epoch=epoch)
            if self.to_break:  # is it time to early stop?
                break
            else:
                if self.lr_scheduler == 'ReduceLROnPlateau': # update lr scheduler if we are using one 
                    if self.args.lr_scheduler_criteria == 'loss':
                        self.lr_scheduler.step(self.val_losses[-1])
                    elif self.args.lr_scheduler_criteria == 'acc': # monitor top-1 acc for lr_scheduler 
                        self.lr_scheduler.step(self.val_topk_accs[1][-1])

            if 5 in self.train_topk_accs:
                epoch_top5_train_acc = self.train_topk_accs[5][-1]
                epoch_top5_val_acc = self.val_topk_accs[5][-1]
            else:
                epoch_top5_train_acc = np.nan
                epoch_top5_val_acc = np.nan
            if 10 in self.train_topk_accs:
                epoch_top10_train_acc = self.train_topk_accs[10][-1]
                epoch_top10_val_acc = self.val_topk_accs[10][-1]
            else:
                epoch_top10_train_acc = np.nan
                epoch_top10_val_acc = np.nan
            logging.info(
                f"\nEnd of epoch: {epoch}, \
                \ntrain loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_topk_accs[1][-1]:.4f}, \
                \ntop-5 train acc: {epoch_top5_train_acc:.4f}, top-10 train acc: {epoch_top10_train_acc:.4f}, \
                \nval loss: {self.val_losses[-1]: .4f}, top-1 val acc: {self.val_topk_accs[1][-1]:.4f}, \
                \ntop-5 val acc: {epoch_top5_val_acc:.4f}, top-10 val acc: {epoch_top10_val_acc:.4f} \
                \n"
            )
            if self.args.lr_floor and self.optimizer.param_groups[0]['lr'] < 2e-6:
                logging.info('Stopping training as learning rate has dropped below 2e-6')
                break 

        logging.info(f'Total training time: {self.stats["train_time"]}')

    def test(self, saved_stats: Optional[dict] = None):
        """
        Evaluates the model on the test set
        Parameters
        ---------
        saved_stats: Optional[dict]
            Test statistics will be stored inside this stats file
            Used to load existing stats file when loading a trained model from checkpoint
        """
        self.model.eval()
        test_loss, test_correct_preds = 0, defaultdict(int)
        test_loader = tqdm(self.test_loader, desc='testing...')
        with torch.no_grad():
            epoch_test_size = 0
            for i, batch in enumerate(test_loader):
                batch_data = batch[0]
                if not isinstance(batch_data, tuple): # not sure what's the purpose of this
                    batch_data = batch_data.to(self.device)
                if self.model_name == 'TransformerEBM':
                    batch_data = (batch_data, 'test')
                batch_mask = batch[1].to(self.device)
                if self.args.prob_file_prefix:
                    batch_probs = batch[3].to(self.device)
                    batch_energies = self._one_batch(
                        batch_data, batch_mask, batch_probs, backprop=False
                    )
                else:
                    batch_energies = self._one_batch(
                        batch_data, batch_mask, backprop=False
                    )
                test_batch_size = batch_energies.shape[0]
                epoch_test_size += test_batch_size

                # for validation/test data, true rxn may not be present!
                # only provide for finetuning step on retro proposals
                if self.args.do_finetune and self.debug:
                    batch_idx = batch[2]
                    batch_true_ranks_array = self.proposals_data['test'][batch_idx, 2].astype('int')
                    batch_true_ranks_valid = batch_true_ranks_array[batch_true_ranks_array < self.args.minibatch_eval]
                    batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                    # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation
                    # (bcos nothing in the numerator, loss is undefined)
                    loss_numerator = batch_energies[
                        np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval], #!= 9999],
                        batch_true_ranks_valid
                    ]
                    loss_denominator = batch_energies[
                        np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval], #!= 9999],
                        :
                    ]
                    batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item() 

                    for k in self.k_to_calc:
                        # index with lowest energy is what the model deems to be the most feasible rxn
                        batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]  
                        batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                        test_correct_preds[k] += batch_correct_preds

                        if k == 1:
                            batch_top1_acc = batch_correct_preds / test_batch_size
                            if self.debug: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                                try:
                                    for j in range(i * self.batch_size, (i+1) * self.batch_size):
                                        if j % (self.test_size // 5) == 0:  # peek at a random sample of current batch to monitor training progress
                                            sample_idx = random.sample(list(range(self.batch_size)), k=1)[0]
                                            sample_true_rank = batch_true_ranks_array[sample_idx] #[0]
                                            sample_pred_rank = batch_preds[sample_idx, 0].item()
                                            sample_true_prod = self.proposals_data['test'][batch_idx[sample_idx], 0]
                                            sample_true_prec = self.proposals_data['test'][batch_idx[sample_idx], 1] 

                                            sample_cand_precs = self.proposals_data['test'][batch_idx[sample_idx], 3:] 
                                            sample_pred_prec = sample_cand_precs[batch_preds[sample_idx]]
                                            sample_orig_prec = sample_cand_precs[0]
                                            logging.info(f'\ntrue product:            {sample_true_prod}')
                                            logging.info(f'pred precursor (rank {sample_pred_rank}): {sample_pred_prec}')
                                            logging.info(f'true precursor (rank {sample_true_rank}): {sample_true_prec}')
                                            logging.info(f'orig precursor (rank 0): {sample_orig_prec}\n')
                                            break
                                except Exception as e:
                                    tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                                    logging.info("".join(tb_str))
                                    logging.info('\nIndex out of range (last minibatch)')
                        elif k == 5:
                            batch_top5_acc = batch_correct_preds / test_batch_size
                        elif k == 10:
                            batch_top10_acc = batch_correct_preds / test_batch_size

                else: # for pre-training step w/ synthetic data, 0-th index is the positive rxn
                    batch_true_ranks = 0
                    batch_loss = (batch_energies[:, 0] + torch.logsumexp(-batch_energies, dim=1)).sum().item() 

                    # calculate top-k acc assuming true index is 0 (for pre-training step)
                    for k in self.k_to_calc:
                        batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1] 
                        batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0] 
                        test_correct_preds[k] += batch_correct_preds

                        if k == 1:
                            batch_top1_acc = batch_correct_preds / test_batch_size
                        elif k == 5:
                            batch_top5_acc = batch_correct_preds / test_batch_size
                        elif k == 10:
                            batch_top10_acc = batch_correct_preds / test_batch_size
                if 5 not in self.k_to_calc:
                    batch_top5_acc = np.nan
                if 10 not in self.k_to_calc:
                    batch_top10_acc = np.nan 
                test_loader.set_description(f"testing...loss={batch_loss / test_batch_size:.4f}, top-1 acc={batch_top1_acc:.4f}, top-5 acc={batch_top5_acc:.4f}, top-10 acc={batch_top10_acc:.4f}")
                test_loader.refresh()
                test_loss += batch_loss   

            for k in self.k_to_calc:
                self.test_topk_accs[k] = test_correct_preds[k] / epoch_test_size # self.test_size

        if saved_stats:
            self.stats = saved_stats
        if len(self.stats.keys()) <= 2:
            raise RuntimeError(
                "self.stats only has 2 keys or less. If loading checkpoint, you need to provide load_stats!"
            )

        self.stats["test_loss"] = test_loss / epoch_test_size # self.test_size 
        logging.info(f'\nTest loss: {self.stats["test_loss"]:.4f}')
        self.stats["test_topk_accs"] = self.test_topk_accs
        for k in self.k_to_calc:
            logging.info(f'Test top-{k} accuracy: {100 * self.stats["test_topk_accs"][k]:.3f}%')

        torch.save(self.stats, self.stats_filename) # override existing train stats w/ train+test stats

    def get_energies_and_loss(
        self,
        phase: str = "test",
        save_energies: Optional[bool] = True,
        name_energies: Optional[Union[str, bytes, os.PathLike]] = None,
        path_to_energies: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> Tuple[Tensor, float]:
        """
        Gets raw energy values from a trained model on a given dataloader 

        Parameters
        ----------
        phase : str (Default = 'test') [train', 'test', 'valid']
            whether to get energies from train/test/val phases
        save_energies : Optional[bool] (Default = True)
            whether to save the energy values to disk
        name_energies : Optional[Union[str, bytes, os.PathLike]] (Default = None)
            name of the energy file to be saved
            if None, defaults to f"{self.model_name}_{self.expt_name}_energies_{phase}.pkl"
        path_to_energies : Optional[Union[str, bytes, os.PathLike]] (Default = None)
            path to folder in which to save the energy outputs
            if None, defaults to path/to/rxnebm/energies 

        Returns
        -------
        energies : Tensor
            energies of shape (# rxns, 1 + # neg rxns)
        loss : float
            the loss value on the provided dataset

        TODO: fix show_neg: index into SMILES molecule vocab to retrieve molecules -->
        save as groups [true product/rct SMILES, 1st NN SMILES, ... K-1'th NN SMILES])
        """
        if phase == "test":
            dataloader = self.test_loader
        elif phase == "train":
            dataloader = self.train_loader
        elif phase == "valid":
            dataloader = self.val_loader

        self.model.eval()
        with torch.no_grad():
            # for training, we know positive rxn is always the 0th-index, even during finetuning
            if self.args.do_finetune and phase != 'train':
                energies_combined, true_ranks = [], []
                loss, epoch_data_size = 0, 0
                for batch in tqdm(dataloader, desc='getting raw energy outputs...'):
                    batch_data = batch[0]
                    if not isinstance(batch_data, tuple):
                        batch_data = batch_data.to(self.device)
                    if self.model_name == 'TransformerEBM':
                        batch_data = (batch_data, phase)
                    batch_mask = batch[1].to(self.device)
                    batch_energies = self._one_batch(
                        batch_data, batch_mask, backprop=False
                    ) 

                    epoch_data_size += batch_energies.shape[0]

                    batch_idx = batch[2]
                    batch_true_ranks_array = self.proposals_data[phase][batch_idx, 2].astype('int')
                    batch_true_ranks_valid = batch_true_ranks_array[batch_true_ranks_array < self.args.minibatch_eval]
                    batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                    # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation (bcos nothing in the numerator)
                    loss_numerator = batch_energies[
                        np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval], #!= 9999],
                        batch_true_ranks_valid
                    ]
                    loss_denominator = batch_energies[np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval], :]
                    batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum()
                    loss += batch_loss

                    energies_combined.append(batch_energies) 
                    true_ranks.append(batch_true_ranks)

                loss /= epoch_data_size
                energies_combined = torch.cat(energies_combined, dim=0).squeeze(dim=-1).cpu() 
                true_ranks = torch.cat(true_ranks, dim=0).squeeze(dim=-1).cpu()
            else: # pre-training
                energies_combined = []
                epoch_data_size = 0
                for batch in tqdm(dataloader, desc='getting raw energy outputs...'):
                    batch_data = batch[0]
                    if not isinstance(batch_data, tuple):
                        batch_data = batch_data.to(self.device)
                    if self.model_name == 'TransformerEBM':
                        batch_data = (batch_data, phase)
                    batch_mask = batch[1].to(self.device)
                    
                    energies = self.model(batch_data)
                    energies = torch.where(batch_mask, energies,
                                        torch.tensor([float('inf')], device=batch_mask.device))
                    energies_combined.append(energies)
                    epoch_data_size += energies.shape[0]

                energies_combined = torch.cat(energies_combined, dim=0).squeeze(dim=-1).cpu() 

                loss = (energies_combined[:, 0] + torch.logsumexp(-1 * energies_combined, dim=1)).sum().item()
                loss /= epoch_data_size # self.val_size
            logging.info(f"\nLoss on {phase} : {loss:.4f}")

        if path_to_energies is None:
            path_to_energies = Path(__file__).resolve().parents[1] / "energies"
        else:
            path_to_energies = Path(path_to_energies)
        if name_energies is None:
            name_energies = f"{self.model_name}_{self.expt_name}_energies_{phase}.pkl"
        if save_energies:
            logging.info(f"Saving energies at: {Path(path_to_energies / name_energies)}")
            torch.save(energies_combined, Path(path_to_energies / name_energies))

        if self.args.do_finetune and phase != 'train':
            self.energies[phase] = energies_combined
            self.true_ranks[phase] = true_ranks.unsqueeze(dim=-1)
            return energies_combined, loss, true_ranks
        else:
            self.stats["train_loss_nodropout"] = loss
            self.energies[phase] = energies_combined
            return energies_combined, loss

    def get_topk_acc(
        self,
        phase: str = "test",
        k: Optional[int] = 1, 
    ) -> Tensor:
        """
        Computes top-k accuracy of trained model in classifying feasible vs infeasible chemical rxns
        (i.e. minimum energy assigned to label 0 of each training sample)

        Parameters
        ----------
        phase : str (Default = 'test') [train', 'test', 'valid']
            whether to get energies from train/test/valid phases
        save_energies: bool (Default = True)
            whether to save the generated energies tensor to disk
        name_energies: Union[str, bytes, os.PathLike] (Default = None)
            filename of energies to save as a .pkl file
            If None, automatically set to 'energies_<phase>_<self.expt_name>'

        Returns
        -------
        energies: tensor
            energies of shape (# rxns, 1 + # neg rxns)

        Also see: self.get_energies_and_loss()
        """
        if self.args.do_finetune and phase != 'train':
            if phase not in self.energies:
                energies, loss, true_ranks = self.get_energies_and_loss(phase=phase)
            if self.energies[phase].shape[1] >= k: 
                pred_labels = torch.topk(self.energies[phase], k=k, dim=1, largest=False)[1]
                topk_accuracy = torch.where(pred_labels == self.true_ranks[phase])[0].shape[0] / pred_labels.shape[0]

                self.stats[f"{phase}_top{k}_acc_nodropout"] = topk_accuracy
                torch.save(self.stats, self.stats_filename)

                logging.info(f"Top-{k} accuracy on {phase} (finetune): {100 * topk_accuracy:.3f}%") 
            else:
                logging.info(f'{k} out of range for dimension 1 on {phase} (finetune)')

        else: # true rank is always 0
            if phase not in self.energies:
                energies, loss = self.get_energies_and_loss(phase=phase)
                self.energies[phase] = energies
                if phase == 'train':
                    self.stats["train_loss_nodropout"] = loss
            
            if self.energies[phase].shape[1] >= k: 
                pred_labels = torch.topk(self.energies[phase], k=k, dim=1, largest=False)[1]
                topk_accuracy = torch.where(pred_labels == 0)[0].shape[0] / pred_labels.shape[0] 

                self.stats[f"{phase}_top{k}_acc_nodropout"] = topk_accuracy
                torch.save(self.stats, self.stats_filename)

                logging.info(f"Top-{k} accuracy on {phase}: {100 * topk_accuracy:.3f}%") 
            else:
                logging.info(f'{k} out of range for dimension 1 on {phase}')


# def accuracy(output, target, topk=(1,)):
# """Computes the accuracy over the k top predictions for the specified values of k"""
# with torch.no_grad():
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res