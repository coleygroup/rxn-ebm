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

from rxnebm.data import dataset, dataset_utils
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

    device : Optional[str] (Default = None) [torch.device('cuda'), torch.device('cpu')]
        device to do training, testing & inference on.
        If None, automatically detects if GPU is available, else uses CPU.
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
        early_stop: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,  
        num_workers: int,
        checkpoint: bool,
        random_seed: int,
        precomp_file_prefix: str,
        checkpoint_folder: Union[str, bytes, os.PathLike],
        expt_name: str,
        augmentations: dict,
        representation: str, 
        lr_scheduler: Optional[str] = None,
        lr_scheduler_factor: Optional[float] = 0.3,
        lr_scheduler_patience: Optional[int] = 1, 
        onthefly: Optional[bool] = False,
        rxn_type: Optional[str] = None,
        fp_type: Optional[str] = None,
        rctfp_size: Optional[int] = None,
        prodfp_size: Optional[int] = None,
        smi_to_fp_dict_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        fp_to_smi_dict_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        mol_fps_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        search_index_filename: Optional[Union[str, bytes, os.PathLike]] = None,
        rxn_smis_file_prefix: Optional[Union[str, bytes, os.PathLike]] = None,
        device: Optional[str] = None,
        distributed: Optional[bool] = False,
        root: Optional[Union[str, bytes, os.PathLike]] = None,
        load_checkpoint: Optional[bool] = False,
        saved_optimizer: Optional[torch.optim.Optimizer] = None,
        saved_stats: Optional[dict] = None,
        begin_epoch: Optional[int] = None,
        **kwargs
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.best_epoch = 0  # will be automatically assigned after 1 epoch
        self.early_stop = early_stop
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_patience = early_stop_patience
        self.num_workers = num_workers
        self.checkpoint = checkpoint
        self.random_seed = random_seed

        self.representation = representation
        ### fingerprint arguments ###
        if self.representation == 'fingerprint':
            self.rxn_type = rxn_type
            self.fp_type = fp_type
            self.rctfp_size = rctfp_size
            self.prodfp_size = prodfp_size
            self.fp_radius = kwargs["fp_radius"]    # just for record keeping purposes

        if root:
            self.root = Path(root)
        else:
            self.root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        self.checkpoint_folder = Path(checkpoint_folder)

        self.expt_name = expt_name
        self.augmentations = augmentations
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
        self.model_name = model.__repr__()
        self.model_args = model_args
        self.distributed = distributed  # TODO: affects how checkpoint is saved
        if saved_optimizer is not None:
            self.optimizer_name = str(optimizer).split(' ')[0]
        else:
            self.optimizer_name = optimizer
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

        self._collate_args()

        if load_checkpoint:
            self._load_checkpoint(saved_optimizer, saved_stats, begin_epoch)
        else:
            self._init_optimizer_and_stats()

        # if onthefly is True, need smi_to_fp_dict_filename, 
        # mol_fps_filename, (if cos: search_index_filename, fp_to_smi_dict_filename)
        if self.representation == 'fingerprint':
            self._init_fp_dataloaders(
                precomp_file_prefix=precomp_file_prefix,
                onthefly=onthefly,
                smi_to_fp_dict_filename=smi_to_fp_dict_filename,
                fp_to_smi_dict_filename=fp_to_smi_dict_filename,
                mol_fps_filename=mol_fps_filename,
                search_index_filename=search_index_filename,
            )
        elif self.representation == "smiles":
            self._init_smi_dataloaders(
                precomp_file_prefix=None,
                onthefly=onthefly,
                smi_to_fp_dict_filename=smi_to_fp_dict_filename,
                fp_to_smi_dict_filename=fp_to_smi_dict_filename,
                mol_fps_filename=mol_fps_filename,
                search_index_filename=search_index_filename,
            )
        self.energies = {}      # for self.get_topk_acc

        model_utils.seed_everything(random_seed)

    def __repr__(self):
        return "Experiment with: " + self.augmentations

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
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "early_stop": self.early_stop,
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
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                optimizer=self.optimizer, mode='min', 
                factor=self.lr_scheduler_factor, patience=self.lr_scheduler_patience, verbose=True
                ) 
        else:
            self.lr_scheduler = None 
        #TODO: lr_scheduler.load_state_dict(checkpoint['scheduler']) https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208/2

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

        self.train_losses = self.stats["train_losses"]
        self.train_accs = self.stats["train_accs"]
        self.val_losses = self.stats["val_losses"]
        self.val_accs = self.stats["val_accs"]
        self.min_val_loss = float("+inf") # need to reset to +inf for _check_earlystop() 
        self.wait = 0  # counter for _check_earlystop()

        if begin_epoch is None:
            raise ValueError("load_checkpoint requires begin_epoch!")
        self.begin_epoch = begin_epoch

    def _init_optimizer_and_stats(self): 
        logging.info("Initialising optimizer & stats...")
        self.optimizer = model_utils.get_optimizer(self.optimizer_name)(self.model.parameters(), lr=self.learning_rate) 
        if self.lr_scheduler_name is not None:
            logging.info(f'Initialising {self.lr_scheduler_name}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                optimizer=self.optimizer, factor=self.lr_scheduler_factor, patience=self.lr_scheduler_patience, verbose=True
                ) 
        else:
            logging.info('Not using any LR Scheduler!')
            self.lr_scheduler = None 

        self.train_losses = []
        self.train_accs = []
        self.min_val_loss = float("+inf")
        self.val_losses = []
        self.val_accs = []
        self.begin_epoch = 0
        self.wait = 0  # counter for _check_earlystop()

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
    ):
        logging.info("Initialising fingerprint dataloaders...")
        if "cos" in self.augmentations:
            worker_init_fn = expt_utils._worker_init_fn_nmslib
        else:
            worker_init_fn = expt_utils._worker_init_fn_default

        rxn_smis_filenames = defaultdict(str)
        precomp_filenames = defaultdict(str)
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

        if self.rxn_type == "sep":
            input_dim = self.rctfp_size + self.prodfp_size
        elif self.rxn_type == "diff":
            input_dim = self.rctfp_size

        pin_memory = True if torch.cuda.is_available() else False
        train_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_filenames["train"],
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
        del train_dataset, val_dataset, test_dataset  # save memory

    def _get_smi_dl(self,
                    phase: str,
                    rxn_smis_file_prefix: str,
                    fp_to_smi_dict_filename: str,
                    smi_to_fp_dict_filename: str,
                    mol_fps_filename: str,
                    mut_smis_filename: str,
                    shuffle: bool = False):
        rxn_smis_filename = f"{rxn_smis_file_prefix}_{phase}.pickle"

        augmented_data = dataset.ReactionDatasetSMILES(
            augmentations=self.augmentations,
            precomp_rxnsmi_filename=None,
            fp_to_smi_dict_filename=fp_to_smi_dict_filename,
            smi_to_fp_dict_filename=smi_to_fp_dict_filename,
            mol_fps_filename=mol_fps_filename,
            rxn_smis_filename=rxn_smis_filename,
            mut_smis_filename=mut_smis_filename,
            onthefly=True,
            seed=self.random_seed
        )

        pin_memory = True if torch.cuda.is_available() else False
        _loader = DataLoader(
            augmented_data,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=dataset_utils.graph_collate_fn_builder(self.device)
        )

        _size = len(augmented_data)
        del augmented_data

        return _loader, _size

    def _init_smi_dataloaders(
        self,
        precomp_file_prefix: Optional[str],
        onthefly: bool,
        fp_to_smi_dict_filename: str,
        smi_to_fp_dict_filename: str,
        mol_fps_filename: str,
        mut_smis_filename: str,
        search_index_filename: str,
        rxn_smis_file_prefix: Optional[str] = None
    ):
        logging.info("Initialising SMILES dataloaders...")
        if onthefly:
            self.train_loader, self.train_size = self._get_smi_dl(
                phase="train",
                rxn_smis_file_prefix=rxn_smis_file_prefix,
                fp_to_smi_dict_filename=fp_to_smi_dict_filename,
                smi_to_fp_dict_filename=smi_to_fp_dict_filename,
                mol_fps_filename=mol_fps_filename,
                mut_smis_filename=mut_smis_filename,
                shuffle=True)
            self.val_loader, self.val_size = self._get_smi_dl(
                phase="valid",
                rxn_smis_file_prefix=rxn_smis_file_prefix,
                fp_to_smi_dict_filename=fp_to_smi_dict_filename,
                smi_to_fp_dict_filename=smi_to_fp_dict_filename,
                mol_fps_filename=mol_fps_filename,
                mut_smis_filename=mut_smis_filename,
                shuffle=False)
            self.test_loader, self.test_size = self._get_smi_dl(
                phase="test",
                rxn_smis_file_prefix=rxn_smis_file_prefix,
                fp_to_smi_dict_filename=fp_to_smi_dict_filename,
                smi_to_fp_dict_filename=smi_to_fp_dict_filename,
                mol_fps_filename=mol_fps_filename,
                mut_smis_filename=mut_smis_filename,
                shuffle=False)
        else:
            raise NotImplementedError

    def _check_earlystop(self, current_epoch):
        self.to_break = 0
        if self.min_val_loss - self.val_losses[-1] < self.early_stop_min_delta:
            if self.early_stop_patience <= self.wait:
                logging.info(
                    f"\nEarly stopped at the end of epoch: {current_epoch}, \
                train loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_accs[-1]:.4f}, \
                \nval loss: {self.val_losses[-1]:.4f}, top-1 val acc: {self.val_accs[-1]:.4f}"
                )
                self.stats["early_stop_epoch"] = current_epoch
                self.to_break = 1  # will break loop
            else:
                self.wait += 1
                logging.info(f"Decrease in val loss < early stop min delta, patience count: {self.wait}")
        else:
            self.wait = 0
            self.min_val_loss = min(self.min_val_loss, self.val_losses[-1])

    def _update_stats(self):
        self.stats["train_time"] = (
            self.stats["train_time"] + (time.time() - self.start) / 60
        )  # in minutes
        self.start = time.time()

        # a list with one value for each epoch
        self.stats["train_losses"] = self.train_losses
        # a list with one value for each epoch
        self.stats["train_accs"] = self.train_accs
        # a list with one value for each epoch
        self.stats["val_losses"] = self.val_losses
        # a list with one value for each epoch
        self.stats["val_accs"] = self.val_accs
        self.stats["min_val_loss"] = self.min_val_loss
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

    def _one_batch(self, batch, mask: Tensor, backprop: bool = True):
        """
        Passes one batch of samples through model to get energies & loss
        Does backprop if training 
        """
        for p in self.model.parameters():
            p.grad = None  # faster, equivalent to self.model.zero_grad()
        energies = self.model(batch)  # size N x K

        # replace all-zero vectors with float('inf'), making those gradients 0 on backprop
        energies = torch.where(mask, energies, torch.tensor([float('inf')], device=mask.device))
 
        # positives are the 0-th index of each sample
        loss = (energies[:, 0] + torch.logsumexp(-energies, dim=1)).sum()

        if backprop:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # index with lowest energy is the most feasible rxn
        pred_labels = torch.topk(energies, 1, dim=1, largest=False)[1]
        # 0-th index should have the lowest energy
        pred_correct = torch.where(pred_labels == 0)[0].shape[0]
        return loss.item(), pred_correct
 
    def train(self):
        self.start = time.time()  # timeit.default_timer()
        for epoch in range(self.begin_epoch, self.epochs):
            self.model.train()
            train_loss, train_correct_preds = 0, 0
            train_loader = tqdm(self.train_loader, desc='training...')
            for batch in train_loader:
                batch_data = batch[0].to(self.device)
                batch_mask = batch[1].to(self.device) 
                curr_batch_loss, curr_batch_correct_preds = self._one_batch(
                    batch_data, batch_mask, backprop=True
                ) 
                train_loader.set_description(f"training...loss={curr_batch_loss/batch[0].shape[0]:.4f}, acc={curr_batch_correct_preds/batch[0].shape[0]:.4f}")
                train_loader.refresh()  
                train_loss += curr_batch_loss 
                train_correct_preds += curr_batch_correct_preds
            self.train_accs.append(train_correct_preds / self.train_size)
            self.train_losses.append(train_loss / self.train_size)

            self.model.eval()
            val_loss, val_correct_preds = 0, 0
            val_loader = tqdm(self.val_loader, desc='validating...')
            for batch in val_loader:
                batch_data = batch[0].to(self.device)
                batch_mask = batch[1].to(self.device)
                curr_batch_loss, curr_batch_correct_preds = self._one_batch(
                    batch_data, batch_mask, backprop=False
                )
                val_loader.set_description(f"validating...loss={curr_batch_loss/batch[0].shape[0]:.4f}, acc={curr_batch_correct_preds/batch[0].shape[0]:.4f}")
                val_loader.refresh()  
                val_loss += curr_batch_loss
                val_correct_preds += curr_batch_correct_preds
            self.val_accs.append(val_correct_preds / self.val_size)
            self.val_losses.append(val_loss / self.val_size)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.val_losses[-1])

            # track best_epoch to facilitate loading of best checkpoint
            if self.val_losses[-1] < self.min_val_loss:
                self.best_epoch = epoch

            self._update_stats()
            if self.checkpoint:
                self._checkpoint_model_and_opt(current_epoch=epoch)
            if self.early_stop:
                self._check_earlystop(current_epoch=epoch)
                if self.to_break:  # is it time to early stop?
                    break

            logging.info(
                f"\nEnd of epoch: {epoch}, \
                \ntrain loss: {self.train_losses[-1]:.6f}, top-1 train acc: {self.train_accs[-1]:.4f}, \
                \nval loss: {self.val_losses[-1]: .6f}, top-1 val acc: {self.val_accs[-1]:.4f}"
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
        test_loss, test_correct_preds = 0, 0
        test_loader = tqdm(self.test_loader, desc='testing...')
        for batch in test_loader:
            batch_data = batch[0].to(self.device)
            batch_mask = batch[1].to(self.device)
            curr_batch_loss, curr_batch_correct_preds = self._one_batch(
                batch_data, batch_mask, backprop=False
            )
            test_loss += curr_batch_loss
            test_correct_preds += curr_batch_correct_preds
            test_loader.set_description(f"testing...loss={curr_batch_loss/batch[0].shape[0]:.4f}, acc={curr_batch_correct_preds/batch[0].shape[0]:.4f}")
            test_loader.refresh() 

        if saved_stats:
            self.stats = saved_stats
        if len(self.stats.keys()) <= 2:
            raise RuntimeError(
                "self.stats only has 2 keys or less. If loading checkpoint, you need to provide load_stats!"
            )

        self.stats["test_loss"] = test_loss / self.test_size
        self.stats["test_acc"] = test_correct_preds / self.test_size
        logging.info(f'Test loss: {self.stats["test_loss"]}')
        logging.info(f'Test top-1 accuracy: {self.stats["test_acc"]}')
        torch.save(self.stats, self.stats_filename) # override existing train stats w/ train+test stats

    def get_energies_and_loss(
        self,
        phase: Optional[str] = "test",
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
        custom_dataloader : Optional[Dataloader] (Default = None)
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
        energies_combined = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='getting raw energy outputs...'):
                batch_data = batch[0].to(self.device)
                batch_mask = batch[1].to(self.device)
                
                energies = self.model(batch_data)
                energies = torch.where(batch_mask, energies,
                                       torch.tensor([float('inf')], device=batch_mask.device))
                energies_combined.append(energies)

            energies_combined = torch.cat(energies_combined, dim=0).squeeze(dim=-1).to(torch.device('cpu'))

            loss = (energies_combined[:, 0] + torch.logsumexp(-1 * energies_combined, dim=1)).sum()
            if phase == "custom":
                loss /= custom_dataset_len
            elif phase == "test":
                loss /= self.test_size
            elif phase == "train":
                loss /= self.train_size
            elif phase == "val":
                loss /= self.val_size
            logging.info(f"Loss on {phase} : {loss.item():.6f}")

        if path_to_energies is None:
            path_to_energies = Path(__file__).resolve().parents[1] / "energies"
        else:
            path_to_energies = Path(path_to_energies)
        if name_energies is None:
            name_energies = f"{self.model_name}_{self.expt_name}_energies_{phase}.pkl"
        if save_energies:
            logging.info(f"Saving energies at: {Path(path_to_energies / name_energies)}")
            torch.save(energies_combined, Path(path_to_energies / name_energies))

        if phase not in self.energies:
            self.energies[phase] = energies_combined
            if phase == 'train':
                self.stats["train_loss_nodropout"] = loss
        return energies_combined, loss

    def get_topk_acc(
        self,
        phase: Optional[str] = "test",
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
        if phase not in self.energies:
            energies, loss = self.get_energies_and_loss(
                phase=phase, custom_dataloader=custom_dataloader
            )
            self.energies[phase] = energies
            if phase == 'train':
                self.stats["train_loss_nodropout"] = loss
        
        if self.energies[phase].shape[1] > k: 
            pred_labels = torch.topk(self.energies[phase], k, dim=1, largest=False)[1]
            topk_accuracy = torch.where(pred_labels == 0)[0].shape[0] / pred_labels.shape[0]

            self.stats[f"{phase}_top_{k}_acc_nodropout"] = topk_accuracy
            torch.save(self.stats, self.stats_filename)

            logging.info(f"Top-{k} accuracy on {phase}: {topk_accuracy}") 
        else:
            logging.info(f'{k} out of range for dimension 1 on {phase}')
