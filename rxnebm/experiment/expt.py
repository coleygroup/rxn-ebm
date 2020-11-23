import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 

from rxnebm.data import dataset
from rxnebm.experiment import expt_utils
from rxnebm.model import model_utils

Tensor = torch.Tensor

"""
NOTE: the way python's imports work, is that I cannot run scripts on their own from that directory
(e.g. I cannot do python FF.py because FF.py imports functions from model.model_utils, and at the time of
executing FF.py, python has no knowledge of this package level information (i.e. __package__ is None))
therefore, I have to run/import all these files/functions from a script in the main directory,
i.e. same folder as trainEBM.py by doing: from model import FF; from experiment import expt etc

To run this script from terminal/interpreter, go to root/of/rxnebm/ then execute python -m rxnebm.experiment.expt
"""


class Experiment:
    """
    NOTE: assumes pre-computed file already exists --> trainEBM.py handles the pre-computation 
    TODO: representation: ['graph', 'string'] 
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
        model: nn.Module,
        model_args: dict,
        batch_size: int,
        epochs: int,
        optimizer: str,
        learning_rate: float,
        num_workers: int,
        checkpoint: bool,
        random_seed: int,
        precomp_file_prefix: str,
        checkpoint_folder: Union[str, bytes, os.PathLike],
        expt_name: str,
        augmentations: dict,
        representation: str, 
        early_stop: bool,
        early_stop_criteria: Optional[str] = 'top1_acc', 
        early_stop_patience: Optional[int] = 1,
        early_stop_min_delta: Optional[float] = 0.05,  
        lr_scheduler: Optional[str] = None,
        lr_scheduler_criteria: Optional[str] = 'acc',
        lr_scheduler_factor: Optional[float] = 0.3,
        lr_scheduler_patience: Optional[int] = 1, 
        proposals_csv_file_prefix: Optional[str] = None,
        onthefly: Optional[bool] = False,
        rxn_smis_file_prefix: Optional[str] = None,
        rxn_type: Optional[str] = None,
        fp_type: Optional[str] = None,
        rctfp_size: Optional[int] = None,
        prodfp_size: Optional[int] = None,
        smi_to_fp_dict_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        fp_to_smi_dict_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        mol_fps_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        search_index_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        debug: Optional[bool] = True, 
        device: Optional[str] = None,
        distributed: Optional[bool] = False,
        root: Optional[Union[str, bytes, os.PathLike]] = None,
        load_checkpoint: Optional[bool] = False,
        saved_optimizer: Optional[torch.optim.Optimizer] = None,
        saved_stats: Optional[dict] = None,
        begin_epoch: Optional[int] = None,
        **kwargs,
    ):
        self.debug = debug
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.best_epoch = 0  # will be automatically assigned after 1 epoch

        self.early_stop = early_stop
        if self.early_stop:
            self.early_stop_criteria = early_stop_criteria
            self.early_stop_min_delta = early_stop_min_delta
            self.early_stop_patience = early_stop_patience

            self.min_val_loss = float("+inf")  
            self.max_val_acc = float("-inf") 
            self.wait = 0  # counter for _check_earlystop()
        else:
            self.early_stop_criteria = None
            self.early_stop_min_delta = None
            self.early_stop_patience = None

        self.num_workers = num_workers
        self.checkpoint = checkpoint
        self.random_seed = random_seed

        if root:
            self.root = Path(root)
        else:
            self.root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        self.checkpoint_folder = Path(checkpoint_folder)

        self.expt_name = expt_name
        self.augmentations = augmentations
        self.representation = representation
        ### fingerprint arguments ###
        if self.representation == 'fingerprint':
            self.rxn_type = rxn_type
            self.fp_type = fp_type
            self.rctfp_size = rctfp_size
            self.prodfp_size = prodfp_size
            self.fp_radius = kwargs["fp_radius"] # just for record keeping purposes

        if self.representation != 'fingerprint' and 'bit' in augmentations.keys():
            raise RuntimeError('Bit Augmentor is only compatible with fingerprint representation!')
        logging.info(f"\nInitialising experiment: {self.expt_name}")
        logging.info(f"Augmentations: {self.augmentations}")

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.model = model
        self.model_name = repr(self.model)
        self.model_args = model_args
        self.distributed = distributed  # TODO: affects how checkpoint is saved

        if saved_optimizer is not None:
            self.optimizer_name = str(optimizer).split(' ')[0]
        else:
            self.optimizer_name = optimizer

        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_criteria = lr_scheduler_criteria
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

        self._collate_args()

        # if onthefly is True, need smi_to_fp_dict_filename, 
        # mol_fps_filename, (if cos: search_index_filename, fp_to_smi_dict_filename)
        if self.representation == 'fingerprint':
            self._init_fp_dataloaders(
                precomp_file_prefix=precomp_file_prefix,
                proposals_csv_file_prefix=proposals_csv_file_prefix,
                onthefly=onthefly,
                smi_to_fp_dict_filename=smi_to_fp_dict_filename,
                fp_to_smi_dict_filename=fp_to_smi_dict_filename,
                mol_fps_filename=mol_fps_filename,
                search_index_filename=search_index_filename,
                rxn_smis_file_prefix=rxn_smis_file_prefix,
            )

        if load_checkpoint:
            self._load_checkpoint(saved_optimizer, saved_stats, begin_epoch)
        else:
            self._init_optimizer_and_stats()
        self.train_losses = []
        self.val_losses = []
        
        if self.representation == 'fingerprint':
            self.maxk = self.train_loader.dataset.data.shape[-1] //  self.train_loader.dataset.input_dim
        elif self.representation == 'smiles':
            self.maxk = len(self.train_loader.dataset._augmented_rxn_smiles[0])
        else:
            raise NotImplementedError('Please add self.maxk for the current representation; \
                                we need maxk to calculate topk_accs up to the maximum allowed value \
                                (otherwise torch will raise index out of range)') 
            # e.g. we can't calculate top-50 acc if we only have 1 positive + 10 negatives per minibatch
        self.train_topk_accs = {}  
        self.val_topk_accs = {}  
        self.test_topk_accs = {} 
        self.k_to_calc = [1, 5, 10, 50] # [1, 2, 3, 5, 10, 20, 50, 100] seems to slow down training...? 
        k_not_to_calc = []
        for k in self.k_to_calc:
            if k > self.maxk:
                k_not_to_calc.append(k)
            else: # init empty lists; 1 value will be appended to each list per epoch 
                self.train_topk_accs[k] = []
                self.val_topk_accs[k] = []  
                self.test_topk_accs[k] = None    
        for k in k_not_to_calc:
            self.k_to_calc.remove(k)
        
        self.energies = {} # for self.get_topk_acc
        self.true_ranks = {} # for self.get_topk_acc (finetuning)

        model_utils.seed_everything(random_seed)

    def __repr__(self):
        return f"Experiment name: {self.expt_name}, \
                with augmentations: {self.augmentations}" 

    def _collate_args(self): 
        if self.representation == 'fingerprint':
            self.fp_args = {
                "rxn_type": self.rxn_type,
                "fp_type": self.fp_type,
                "rctfp_size": self.rctfp_size,
                "prodfp_size": self.prodfp_size,
                "fp_radius": self.fp_radius,
            }
        self.train_args = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler_name,            
            "lr_scheduler_criteria": self.lr_scheduler_criteria,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_scheduler_patience": self.lr_scheduler_patience,
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
        if self.lr_scheduler_name is not None:
            logging.info(f'Initialising {self.lr_scheduler_name}')
            if self.lr_scheduler_criteria == 'acc':
                mode = 'max'
            elif self.lr_scheduler_criteria == 'loss':
                mode = 'min'
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                optimizer=self.optimizer, mode=mode, 
                factor=self.lr_scheduler_factor, patience=self.lr_scheduler_patience, verbose=True
                ) 
        else:
            self.lr_scheduler = None 
        #NOTE/TODO: lr_scheduler.load_state_dict(checkpoint['scheduler']) https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208/2

        if saved_stats is None:
            raise ValueError("load_checkpoint requires saved_stats!")
        self.stats = saved_stats
        self.stats_filename = (
            self.checkpoint_folder / f"{self.model_name}_{self.expt_name}_stats.pkl"
        )

        self.model_args = self.stats["model_args"]
        self.fp_args = self.stats["fp_args"]
        self.train_args = self.stats["train_args"]
        self.augmentations = self.stats["augmentations"]

        if begin_epoch is None:
            raise ValueError("load_checkpoint requires begin_epoch!")
        else:
            self.begin_epoch = begin_epoch

    def _init_optimizer_and_stats(self): 
        logging.info("Initialising optimizer & stats...")
        self.optimizer = model_utils.get_optimizer(self.optimizer_name)(self.model.parameters(), lr=self.learning_rate) 
        if self.lr_scheduler_name is not None:
            logging.info(f'Initialising {self.lr_scheduler_name}')
            if self.lr_scheduler_criteria == 'acc':
                mode = 'max'
            elif self.lr_scheduler_criteria == 'loss':
                mode = 'min'
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                optimizer=self.optimizer, mode=mode, 
                factor=self.lr_scheduler_factor, patience=self.lr_scheduler_patience, verbose=True
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
        rxn_smis_file_prefix: Optional[str] = None,
        proposals_csv_file_prefix: Optional[str] = None,
    ):
        logging.info("Initialising dataloaders...")
        if "cos" in self.augmentations:
            worker_init_fn = expt_utils._worker_init_fn_nmslib
        else:
            worker_init_fn = expt_utils._worker_init_fn_default

        rxn_smis_filenames = defaultdict(str)
        precomp_filenames = defaultdict(str)
        proposals_csv_filenames = defaultdict(str)
        if onthefly:
            augmented_data = dataset.AugmentedDataFingerprints(
                self.augmentations,
                smi_to_fp_dict_filename,
                fp_to_smi_dict_filename,
                mol_fps_filename,
                search_index_filename,
            )
            for phase in [
                "train",
                "valid",
                "test",
            ]:  # rxn_smis_file_prefix = '50k_clean_rxnsmi_noreagent'
                rxn_smis_filenames[phase] = rxn_smis_file_prefix + f"_{phase}.pickle"
        else:
            augmented_data = None
            for phase in [
                "train",
                "valid",
                "test",
            ]:  # precomp_file_prefix = '50k_count_rdm_5'
                precomp_filenames[phase] = precomp_file_prefix + f"_{phase}.npz"
        
        if proposals_csv_file_prefix is not None:
            for phase in ['valid', 'test']:
                proposals_csv_filenames[phase] = proposals_csv_file_prefix + f"_{phase}.csv"

        if self.rxn_type == "sep":
            input_dim = self.rctfp_size + self.prodfp_size
        elif self.rxn_type == "diff":
            input_dim = self.rctfp_size

        pin_memory = True if torch.cuda.is_available() else False
        train_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_filenames["train"],
            proposals_csv_filename=proposals_csv_filenames["train"],
            rxn_smis_filename=rxn_smis_filenames["train"],
            onthefly=onthefly,
            augmented_data=augmented_data,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=True,
            pin_memory=pin_memory,
        )
        self.train_size = len(train_dataset)

        val_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_filenames["valid"],
            proposals_csv_filename=proposals_csv_filenames["valid"],
            rxn_smis_filename=rxn_smis_filenames["valid"],
            onthefly=onthefly,
            augmented_data=augmented_data,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=pin_memory,
        )
        self.val_size = len(val_dataset)

        test_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_filenames["test"],
            proposals_csv_filename=proposals_csv_filenames["test"],
            rxn_smis_filename=rxn_smis_filenames["test"],
            onthefly=onthefly,
            augmented_data=augmented_data,
        )
        self.test_loader = DataLoader(
            test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=pin_memory,
        )
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset  

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
                    \nval loss: {self.val_losses[-1]:.4f}, top-1 val acc: {self.val_topk_accs[1][-1]:.4f}"
                    )
                    self.stats["early_stop_epoch"] = current_epoch
                    self.to_break = 1  # will break loop
                else:
                    self.wait += 1
                    logging.info(
                        f'\nIncrease in top-{k} val acc < early stop min delta {self.early_stop_min_delta}, \
                        \npatience count: {self.wait}')
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

    def _one_batch(self, batch: Tensor, mask: Tensor, backprop: bool = True):
        """
        Passes one batch of samples through model to get energies & loss
        Does backprop if training 
        """
        for p in self.model.parameters():
            p.grad = None  # faster, equivalent to self.model.zero_grad()
        energies = self.model(batch)  # size N x K

        # replace all-zero vectors with float('inf'), making those gradients 0 on backprop
        energies = torch.where(mask, energies, torch.tensor([float('inf')], device=mask.device))

        if backprop:
            # for training only: positives are the 0-th index of each minibatch (row)
            loss = (energies[:, 0] + torch.logsumexp(-energies, dim=1)).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item(), energies.cpu().detach() 
        else: # if fine-tuning: validation/testing, cannot calculate loss naively, need the true_ranks of precursors from retrosynthesis models
            return energies.cpu().detach() 
 
    def train(self):
        self.start = time.time()  # timeit.default_timer()
        self.to_break = 0
        for epoch in range(self.begin_epoch, self.epochs):
            self.model.train()
            train_loss, train_correct_preds = 0, defaultdict(int)
            train_loader = tqdm(self.train_loader, desc='training...')
            for batch in train_loader: 
                batch_data = batch[0].to(self.device)
                batch_mask = batch[1].to(self.device) 
                batch_loss, batch_energies = self._one_batch(
                    batch_data, batch_mask, backprop=True
                ) 
                train_loss += batch_loss

                for k in self.k_to_calc: # various top-k accuracies
                    # index with lowest energy is what the model deems to be the most feasible rxn 
                    batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1] 
                    # for training, true rxn is always at 0th-index 
                    batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0]
                    train_correct_preds[k] += batch_correct_preds
                    if k == 1:
                        batch_top1_acc = batch_correct_preds/batch[0].shape[0]
                    elif k == 10:
                        batch_top10_acc = batch_correct_preds/batch[0].shape[0]
                if 10 not in self.k_to_calc:
                    batch_top10_acc = np.nan 
                train_loader.set_description(f"training...loss={batch_loss/batch[0].shape[0]:.4f}, top-1 acc={batch_top1_acc:.4f}, top-10 acc={batch_top10_acc:.4f}")
                train_loader.refresh()
            
            for k in self.k_to_calc:     
                self.train_topk_accs[k].append( train_correct_preds[k] / self.train_size ) 
            self.train_losses.append(train_loss / self.train_size)

            # validation
            self.model.eval()
            with torch.no_grad(): 
                val_loss, val_correct_preds = 0, defaultdict(int)
                val_loader = tqdm(self.val_loader, desc='validating...')
                for i, batch in enumerate(val_loader):
                    batch_data = batch[0].to(self.device)
                    batch_mask = batch[1].to(self.device)
                    batch_energies = self._one_batch(
                        batch_data, batch_mask, backprop=False
                    )   

                    # for validation/test data, true rxn may not be present! 
                    if self.val_loader.dataset.proposals_data is not None: # only provide for finetuning step on retro proposals
                        batch_idx = batch[2]
                        batch_true_ranks_array = self.val_loader.dataset.proposals_data[batch_idx, 2].astype('int')
                        batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                        # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation (bcos nothing in the numerator, loss is undefined)
                        loss_numerator = batch_energies[
                                                                np.arange(batch_energies.shape[0])[ batch_true_ranks_array != 9999 ], 
                                                                batch_true_ranks_array[ batch_true_ranks_array != 9999 ]
                                                            ]
                        loss_denominator = batch_energies[
                                                                np.arange(batch_energies.shape[0])[ batch_true_ranks_array != 9999 ], 
                                                                :
                                                            ]
                        batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item() 

                        for k in self.k_to_calc:
                            # index with lowest energy is what the model deems to be the most feasible rxn
                            batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]  
                            batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                            val_correct_preds[k] += batch_correct_preds

                            if k == 1:
                                batch_top1_acc = batch_correct_preds/batch[0].shape[0]
                        
                                if self.debug: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                                    try:
                                        for j in range(i * self.batch_size, (i+1) * self.batch_size):
                                            if j % (self.val_size // 5) == 0:  # peek at a random sample of current batch to monitor training progress
                                                sample_idx = random.sample( list(range(self.batch_size)), k=1 )
                                                sample_true_rank = batch_true_ranks_array[sample_idx][0]
                                                sample_pred_rank = batch_preds[sample_idx, 0].item() 
                                                sample_true_prod = self.val_loader.dataset.proposals_data[ batch_idx[sample_idx], 0 ]
                                                sample_true_precursor = self.val_loader.dataset.proposals_data[ batch_idx[sample_idx], 1 ]  

                                                sample_cand_precursors = self.val_loader.dataset.proposals_data[ batch_idx[sample_idx], 3: ] 
                                                sample_pred_precursor = sample_cand_precursors[ batch_preds[sample_idx] ]
                                                logging.info(f'\ntrue product:   {sample_true_prod}')
                                                logging.info(f'pred precursor (rank {sample_pred_rank}): {sample_pred_precursor}')
                                                logging.info(f'true precursor (rank {sample_true_rank}): {sample_true_precursor}')
                                                break
                                    except: # do nothing 
                                        logging.info('\nIndex out of range (last minibatch)') 
                            elif k == 10:
                                batch_top10_acc = batch_correct_preds/batch[0].shape[0]                       
                    
                    else: # for pre-training step w/ synthetic data, 0-th index is the positive rxn
                        batch_true_ranks = 0
                        batch_loss = (batch_energies[:, 0] + torch.logsumexp(-batch_energies, dim=1)).sum().item() 

                        # calculate top-k acc assuming true index is 0 (for pre-training step)
                        for k in self.k_to_calc:
                            batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1] 
                            batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0] 
                            val_correct_preds[k] += batch_correct_preds

                            if k == 1:
                                batch_top1_acc = batch_correct_preds/batch[0].shape[0]
                            elif k == 10:
                                batch_top10_acc = batch_correct_preds/batch[0].shape[0]
                    if 10 not in self.k_to_calc:
                        batch_top10_acc = np.nan 
                    val_loader.set_description(f"training...loss={batch_loss/batch[0].shape[0]:.4f}, top-1 acc={batch_top1_acc:.4f}, top-10 acc={batch_top10_acc:.4f}")
                    val_loader.refresh() 
                    
                    val_loss += batch_loss  
                
                for k in self.k_to_calc:
                    self.val_topk_accs[k].append( val_correct_preds[k] / self.val_size )
                self.val_losses.append(val_loss / self.val_size)

            # track best_epoch to facilitate loading of best checkpoint
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
                if self.lr_scheduler is not None: # update lr scheduler if we are using one 
                    if self.lr_scheduler_criteria == 'loss':
                        self.lr_scheduler.step(self.val_losses[-1])
                    elif self.lr_scheduler_criteria == 'acc': # monitor top-1 acc for lr_scheduler 
                        self.lr_scheduler.step(self.val_topk_accs[1][-1])

            if 10 in self.train_topk_accs:
                epoch_top10_train_acc = self.train_topk_accs[10][-1]
                epoch_top10_val_acc = self.val_topk_accs[10][-1]
            else:
                epoch_top10_train_acc = np.nan
                epoch_top10_val_acc = np.nan
            logging.info(
                f"\nEnd of epoch: {epoch}, \
                \ntrain loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_topk_accs[1][-1]:.4f}, \
                \ntop-10 train acc: {epoch_top10_train_acc:.4f}, \
                \nval loss: {self.val_losses[-1]: .4f}, top-1 val acc: {self.val_topk_accs[1][-1]:.4f}, \
                \ntop-10 val acc: {epoch_top10_val_acc:.4f}"
            )

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
            for i, batch in enumerate(test_loader):
                batch_data = batch[0].to(self.device)
                batch_mask = batch[1].to(self.device)
                batch_energies = self._one_batch(
                    batch_data, batch_mask, backprop=False
                ) 
                # for validation/test data, true rxn may not be present! 
                if self.test_loader.dataset.proposals_data is not None: # only provide for finetuning step on retro proposals
                    batch_idx = batch[2]
                    batch_true_ranks_array = self.test_loader.dataset.proposals_data[batch_idx, 2].astype('int')  
                    batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                    # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation (bcos nothing in the numerator, loss is undefined)
                    loss_numerator = batch_energies[
                                                            np.arange(batch_energies.shape[0])[batch_true_ranks_array != 9999], 
                                                            batch_true_ranks_array[batch_true_ranks_array != 9999]
                                                        ]
                    loss_denominator = batch_energies[
                                                            np.arange(batch_energies.shape[0])[batch_true_ranks_array != 9999], 
                                                            :
                                                        ]
                    batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item() 

                    for k in self.k_to_calc:
                        # index with lowest energy is what the model deems to be the most feasible rxn
                        batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]  
                        batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                        test_correct_preds[k] += batch_correct_preds

                        if k == 1:
                            batch_top1_acc = batch_correct_preds/batch[0].shape[0]

                            if self.debug: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                                try:
                                    for j in range(i * self.batch_size, (i+1) * self.batch_size):
                                        if j % (self.test_size // 5) == 0:  # peek at a random sample of current batch to monitor training progress
                                            sample_idx = random.sample(list(range(self.batch_size)), k=1)
                                            sample_true_rank = batch_true_ranks_array[sample_idx][0]
                                            sample_pred_rank = batch_preds[sample_idx, 0].item()
                                            sample_true_prod = self.test_loader.dataset.proposals_data[batch_idx[sample_idx], 0]
                                            sample_true_precursor = self.test_loader.dataset.proposals_data[batch_idx[sample_idx], 1] 

                                            sample_cand_precursors = self.test_loader.dataset.proposals_data[batch_idx[sample_idx], 3:] 
                                            sample_pred_precursor = sample_cand_precursors[batch_preds[sample_idx]]
                                            logging.info(f'\ntrue product:   {sample_true_prod}')
                                            logging.info(f'pred precursor (rank {sample_pred_rank}): {sample_pred_precursor}')
                                            logging.info(f'true precursor (rank {sample_true_rank}): {sample_true_precursor}\n')
                                            break
                                except:
                                    logging.info('\nIndex out of range (last minibatch)')
                        elif k == 10:
                            batch_top10_acc = batch_correct_preds/batch[0].shape[0]

                else: # for pre-training step w/ synthetic data, 0-th index is the positive rxn
                    batch_true_ranks = 0
                    batch_loss = (batch_energies[:, 0] + torch.logsumexp(-batch_energies, dim=1)).sum().item() 

                    # calculate top-k acc assuming true index is 0 (for pre-training step)
                    for k in self.k_to_calc:
                        batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1] 
                        batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0] 
                        test_correct_preds[k] += batch_correct_preds

                        if k == 1:
                            batch_top1_acc = batch_correct_preds/batch[0].shape[0]
                        elif k == 10:
                            batch_top10_acc = batch_correct_preds/batch[0].shape[0]
                if 10 not in self.k_to_calc:
                    batch_top10_acc = np.nan 
                test_loader.set_description(f"training...loss={batch_loss/batch[0].shape[0]:.4f}, top-1 acc={batch_top1_acc:.4f}, top-10 acc={batch_top10_acc:.4f}")
                test_loader.refresh() 

                test_loss += batch_loss   

            for k in self.k_to_calc:
                self.test_topk_accs[k] = test_correct_preds[k] / self.test_size

        if saved_stats:
            self.stats = saved_stats
        if len(self.stats.keys()) <= 2:
            raise RuntimeError(
                "self.stats only has 2 keys or less. If loading checkpoint, you need to provide load_stats!"
            )

        self.stats["test_loss"] = test_loss / self.test_size 
        logging.info(f'\nTest loss: {self.stats["test_loss"]:.4f}')

        self.stats["test_topk_accs"] = self.test_topk_accs
        for k in self.k_to_calc:
            logging.info(f'Test top-{k} accuracy: {100 * self.stats["test_topk_accs"][k]:.3f}%')

        torch.save(self.stats, self.stats_filename) # override existing train stats w/ train+test stats

    def get_energies_and_loss(
        self,
        phase: str = "test",
        finetune: bool = False, 
        custom_dataloader: Optional[torch.utils.data.DataLoader] = None,
        save_energies: Optional[bool] = True,
        name_energies: Optional[Union[str, bytes, os.PathLike]] = None,
        path_to_energies: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> Tuple[Tensor, float]:
        """
        Gets raw energy values from a trained model on a given dataloader 

        Parameters
        ----------
        phase : str (Default = 'test') [train', 'test', 'val', 'custom']
            whether to get energies from train/test/val phases or a custom_dataloader
        finetune : bool (Default = False)
            whether finetuning a pre-trained EBM or training from scratch on synthetic data
            affects calculation of loss & top-k accuracy! 
        custom_dataloader : Optional[Dataloader] (Default = None) (unused for now; in case we want to evaluate on some external test set)
            custom dataloader that loops through dataset that is not the original train, test or val
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
        if phase == "custom":
            custom_dataset_len = len(custom_dataloader.dataset)
            dataloader = custom_dataloader
        elif phase == "test":
            dataloader = self.test_loader
        elif phase == "train":
            dataloader = self.train_loader
        elif phase == "val":
            dataloader = self.val_loader

        self.model.eval()
        with torch.no_grad():
            if finetune and phase != 'train': # for training, we know positive rxn is always the 0th-index, even during finetuning
                energies_combined, true_ranks = [], []
                loss = 0
                for batch in tqdm(dataloader, desc='getting raw energy outputs...'):
                    batch_data = batch[0].to(self.device)
                    batch_mask = batch[1].to(self.device)
                    batch_energies = self._one_batch(
                        batch_data, batch_mask, backprop=False
                    ) 

                    batch_idx = batch[2]
                    batch_true_ranks_array = dataloader.dataset.proposals_data[batch_idx, 2].astype('int') 
                    batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                    # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation (bcos nothing in the numerator)
                    loss_numerator = batch_energies[
                                                            np.arange(batch_energies.shape[0])[batch_true_ranks_array != 9999], 
                                                            batch_true_ranks[batch_true_ranks != 9999]
                                                        ]
                    loss_denominator = batch_energies[np.arange(batch_energies.shape[0])[batch_true_ranks_array != 9999], :]
                    batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum()
                    loss += batch_loss

                    energies_combined.append(batch_energies) 
                    true_ranks.append(batch_true_ranks)

                energies_combined = torch.cat(energies_combined, dim=0).squeeze(dim=-1).to(torch.device('cpu')) 
                true_ranks = torch.cat(true_ranks, dim=0).squeeze(dim=-1).to(torch.device('cpu')) 

            else: # pre-training
                energies_combined = [] 
                for batch in tqdm(dataloader, desc='getting raw energy outputs...'):
                    batch_data = batch[0].to(self.device)
                    batch_mask = batch[1].to(self.device)
                    
                    energies = self.model(batch_data)
                    energies = torch.where(batch_mask, energies,
                                        torch.tensor([float('inf')], device=batch_mask.device))
                    energies_combined.append(energies)

                energies_combined = torch.cat(energies_combined, dim=0).squeeze(dim=-1).to(torch.device('cpu'))

                loss = (energies_combined[:, 0] + torch.logsumexp(-1 * energies_combined, dim=1)).sum().item() 
            
            if phase == "custom":
                loss /= custom_dataset_len
            elif phase == "test":
                loss /= self.test_size
            elif phase == "train":
                loss /= self.train_size
            elif phase == "val":
                loss /= self.val_size
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
 
        if finetune and phase != 'train':
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
        finetune: bool = False,
        custom_dataloader: Optional[torch.utils.data.DataLoader] = None,
        k: Optional[int] = 1, 
    ) -> Tensor:
        """
        Computes top-k accuracy of trained model in classifying feasible vs infeasible chemical rxns
        (i.e. minimum energy assigned to label 0 of each training sample)

        Parameters
        ----------
        phase : str (Default = 'test') [train', 'test', 'val', 'custom']
            whether to get energies from train/test/valid phases or a custom_dataloader
        finetune : bool (Default = False)
            whether finetuning a pre-trained EBM or training from scratch on synthetic data
            affects calculation of loss & top-k accuracy! 
        custom_dataloader : Optional[Dataloader] (Default = None)
            custom dataloader that loops through dataset that is not the original train, test or val
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
        if custom_dataloader is not None:
            phase = 'custom'

        if finetune and phase != 'train':
            if phase not in self.energies:
                energies, loss, true_ranks = self.get_energies_and_loss(
                    phase=phase, finetune=True, custom_dataloader=custom_dataloader
                ) 
            
            if self.energies[phase].shape[1] >= k: 
                pred_labels = torch.topk(self.energies[phase], k=k, dim=1, largest=False)[1]
                topk_accuracy = torch.where(pred_labels == self.true_ranks[phase])[0].shape[0] / pred_labels.shape[0]

                self.stats[f"{phase}_top{k}_acc_nodropout"] = topk_accuracy
                torch.save(self.stats, self.stats_filename)

                logging.info(f"Top-{k} accuracy on {phase} (finetune): {100 * topk_accuracy:.3f}%") 
            else:
                logging.info(f'{k} out of range for dimension 1 on {phase} (finetune')

        else: # true rank is always 0
            if phase not in self.energies:
                energies, loss = self.get_energies_and_loss(
                    phase=phase, finetune=False, custom_dataloader=custom_dataloader
                )
                self.energies[phase] = energies
                if phase == 'train':
                    self.stats["train_loss_nodropout"] = loss
            
            if self.energies[phase].shape[1] >= k: 
                pred_labels = torch.topk(self.energies[phase], k, dim=1, largest=False)[1]
                topk_accuracy = torch.where(pred_labels == 0)[0].shape[0] / pred_labels.shape[0] 

                self.stats[f"{phase}_top{k}_acc_nodropout"] = topk_accuracy
                torch.save(self.stats, self.stats_filename)

                logging.info(f"Top-{k} accuracy on {phase}: {100 * topk_accuracy:.3f}%") 
            else:
                logging.info(f'{k} out of range for dimension 1 on {phase}')
