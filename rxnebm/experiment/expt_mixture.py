import argparse
import logging
import math
import traceback
import os
import random
import time
from functools import partial
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 

from rxnebm.data import dataset, dataset_utils
from rxnebm.experiment import expt_utils
from rxnebm.model import model_utils

Tensor = torch.Tensor
try:
    send_message = partial(expt_utils.send_message, 
                       chat_id=os.environ['CHAT_ID'], 
                       bot_token=os.environ['BOT_TOKEN'])
except Exception as e:
    pass

class Experiment:
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        model_name: str,
        model_args: dict,
        augmentations: dict,
        onthefly: Optional[bool] = False,
        debug: Optional[bool] = True, 
        gpu: Optional[str] = None,
        dataparallel: Optional[bool] = False,
        root: Optional[Union[str, bytes, os.PathLike]] = None,
        load_checkpoint: Optional[bool] = False,
        saved_optimizer: Optional[torch.optim.Optimizer] = None,
        saved_stats: Optional[dict] = None,
        begin_epoch: Optional[int] = None,
        vocab: Dict[str, int] = None,
    ):
        self.args = args
        self.debug = debug
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.best_epoch = 0  # will be tracked during training
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
            self.early_stop_criteria = args.early_stop_criteria # still needed to track best_epoch
            self.early_stop_min_delta = None
            self.early_stop_patience = None
            self.min_val_loss = float("+inf") # still needed to track best_epoch
            self.max_val_acc = float("-inf") # still needed to track best_epoch
            self.wait = None

        self.num_workers = args.num_workers
        self.checkpoint = args.checkpoint
        self.checkpoint_every = args.checkpoint_every
        self.random_seed = args.random_seed

        if root:
            self.root = Path(root)
        else:
            self.root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
        self.checkpoint_folder = Path(self.args.checkpoint_folder)

        self.expt_name = args.expt_name
        self.augmentations = None
        self.representation = args.representation
        # if self.representation != 'fingerprint' and 'bit' in augmentations:
        #     raise RuntimeError('Bit Augmentor is only compatible with fingerprint representation!')
        logging.info(f"\nInitialising experiment: {self.expt_name}")
        # logging.info(f"Augmentations: {self.augmentations}")

        if gpu is not None: # doing DistributedDataParallel training
            rank = args.nr * args.gpus + gpu
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
            self.gpu = gpu
            self.rank = rank
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(self.device)
            self.gpu = None
            self.rank = None
        self.model = model
        self.model_name = model_name
        self.model_args = model_args
        self.dataparallel = dataparallel # affects how checkpoint is saved & how weights should be loaded

        if saved_optimizer is not None:
            self.optimizer_name = str(self.args.optimizer).split(' ')[0]
        else:
            self.optimizer_name = args.optimizer
        self.lr_scheduler_name = args.lr_scheduler

        self._collate_args()

        self.proposals_data = {}
        self.proposals_csv_filenames = defaultdict(str)
        for phase in ['train', 'valid', 'test']:
            self.proposals_csv_filenames[phase] = self.args.proposals_csv_file_prefix + f"_{phase}.csv"
            if phase != 'train': # this is just for visualization purpose during val/test
                self.proposals_data[phase] = pd.read_csv(self.root / self.proposals_csv_filenames[phase], index_col=None, dtype='str').values

        if self.representation == 'fingerprint':
            self._init_fp_dataloaders(
                prodfps_file_prefix=self.args.prodfps_file_prefix,
                labels_file_prefix=self.args.labels_file_prefix
            )
        else:
            raise ValueError('Only product fingerprints are accepted')

        if load_checkpoint:
            self._load_checkpoint(saved_optimizer, saved_stats, begin_epoch)
        else:
            self._init_optimizer_and_stats()

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.test_accs = []
        
        self.preds = {}

    def __repr__(self):
        return f"Experiment name: {self.expt_name}"

    def _collate_args(self):
        self.train_args = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler_name,
            "early_stop": self.early_stop,
            "early_stop_patience": self.early_stop_patience,
            "random_seed": self.random_seed,
            "expt_name": self.expt_name,
            "device": self.device,
            "model_name": self.model_name,
            "optimizer": self.optimizer_name,
            "args": self.args,
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
        for state in saved_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(self.gpu)
        self.optimizer = saved_optimizer  # load optimizer w/ state dict from checkpoint
        if self.args.new_lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.new_lr
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
                                        cooldown=self.args.lr_cooldown,
                                        verbose=True
                                        ) 
        elif self.lr_scheduler_name == 'CosineAnnealingWarmRestarts':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer, 
                                        T_0=self.args.lr_scheduler_T_0,
                                        eta_min=self.args.lr_min,
                                        )
        elif self.lr_scheduler_name == 'OneCycleLR':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer,
                                        max_lr=self.learning_rate,
                                        epochs=self.epochs,
                                        steps_per_epoch=self.train_size // self.batch_size,
                                        last_epoch=self.args.lr_scheduler_last_batch
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
            "train_args": self.train_args, # new, provided as args
            "model_args": self.model_args,
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
            else:
                raise ValueError(f'Unsupported lr_scheduler_criteria {self.args.lr_scheduler_criteria}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer,
                                        mode=mode,
                                        factor=self.args.lr_scheduler_factor,
                                        patience=self.args.lr_scheduler_patience,
                                        cooldown=self.args.lr_cooldown,
                                        verbose=True
                                        ) 
        elif self.lr_scheduler_name == 'CosineAnnealingWarmRestarts':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer, 
                                        T_0=self.args.lr_scheduler_T_0,
                                        eta_min=self.args.lr_min,
                                        )
        elif self.lr_scheduler_name == 'OneCycleLR':
            logging.info(f'Initialising {self.lr_scheduler_name}')
            self.lr_scheduler = model_utils.get_lr_scheduler(self.lr_scheduler_name)(
                                        optimizer=self.optimizer,
                                        max_lr=self.learning_rate,
                                        epochs=self.epochs,
                                        steps_per_epoch=self.train_size // self.batch_size,
                                        last_epoch=self.args.lr_scheduler_last_batch
                                        )
        else:
            logging.info('Not using any LR Scheduler!')
            self.lr_scheduler = None 

        self.begin_epoch = 0 
        self.stats = {
            "train_args": self.train_args,
            "model_args": self.model_args,
            "augmentations": self.augmentations, # None
            "train_time": 0,
        }
        self.stats_filename = (
            self.checkpoint_folder / f"{self.model_name}_{self.expt_name}_stats.pkl"
        )

    def _init_fp_dataloaders(
        self,
        prodfps_file_prefix: str,
        labels_file_prefix: str
    ):
        logging.info("Initialising Mixture Fingerprint dataloaders...")
        train_dataset = dataset.MixtureDatasetFingerprints(
            prodfps_filename=f"{prodfps_file_prefix}_train.npz",
            labels_filename=f"{labels_file_prefix}_train.npy"
        )
        if self.gpu is not None: # train_sampler is needed in self.train_distributed() so we have to save it as an attribute
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.args.world_size,
                rank=self.rank
            )
        else:
            self.train_sampler = None
        self.train_loader = DataLoader(
            train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if self.train_sampler is None else False,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last,
            sampler=self.train_sampler
        )
        self.train_size = len(train_dataset)

        val_dataset = dataset.MixtureDatasetFingerprints(
            prodfps_filename=f"{prodfps_file_prefix}_valid.npz",
            labels_filename=f"{labels_file_prefix}_valid.npy"
        )
        if self.gpu is not None:
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.args.world_size,
                rank=self.rank
            )
        else:
            self.val_sampler = None
        self.val_loader = DataLoader(
            val_dataset,
            self.args.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last,
            sampler=self.val_sampler
        )
        self.val_size = len(val_dataset)

        test_dataset = dataset.MixtureDatasetFingerprints(
            prodfps_filename=f"{prodfps_file_prefix}_test.npz",
            labels_filename=f"{labels_file_prefix}_test.npy"
        )
        if self.gpu is not None:
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=self.args.world_size,
                rank=self.rank
            )
        else:
            self.test_sampler = None
        self.test_loader = DataLoader(
            test_dataset,
            self.args.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last,
            sampler=self.test_sampler
        )
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset

    def _get_smi_dl(self, phase: str, shuffle: bool = False):
        raise NotImplementedError

    def _init_smi_dataloaders(self, onthefly: bool):
        raise NotImplementedError

    def _check_earlystop(self, current_epoch):
        if self.early_stop_criteria == 'loss': 
            if self.min_val_loss - self.val_losses[-1] < self.early_stop_min_delta:
                if self.early_stop_patience <= self.wait:
                    logging.info(
                        f"\nEarly stopped at the end of epoch: {current_epoch}, \
                    \ntrain loss: {self.train_losses[-1]:.4f}, train acc: {self.train_accs[-1]:.4f}, \
                    \nval loss: {self.val_losses[-1]:.4f}, val acc: {self.val_accs[-1]:.4f} \
                    ")
                    self.stats["early_stop_epoch"] = current_epoch
                    self.to_break = 1  # will break loop
                else:
                    self.wait += 1
                    logging.info(f"\nDecrease in val loss < early stop min delta {self.early_stop_min_delta}, \
                                patience count: {self.wait}")
            else:
                self.wait = 0
                self.min_val_loss = min(self.min_val_loss, self.val_losses[-1])

        elif self.early_stop_criteria == 'acc':
            if self.max_val_acc - self.val_accs[-1] > self.early_stop_min_delta:
                if self.early_stop_patience <= self.wait:
                    message = f"\nEarly stopped at the end of epoch: {current_epoch}, \
                    \ntrain loss: {self.train_losses[-1]:.4f}, train acc: {self.train_accs[-1]:.4f}, \
                    \nval loss: {self.val_losses[-1]:.4f}, val acc: {self.val_accs[-1]:.4f} \
                    \n"
                    logging.info(message)
                    if self.rank == 0 or self.rank is None:
                        try:
                            message += f'{self.expt_name}'
                            send_message(message)
                        except Exception as e:
                            pass
                    self.stats["early_stop_epoch"] = current_epoch
                    self.to_break = 1  # will break loop
                else:
                    self.wait += 1
                    logging.info(
                        f'\nIncrease in val acc < early stop min delta {self.early_stop_min_delta}, \
                        \npatience count: {self.wait} \
                        \n')
            else:
                self.wait = 0
                self.max_val_acc = max(self.max_val_acc, self.val_accs[-1])

    def _update_stats(self):
        self.stats["train_time"] = (
            self.stats["train_time"] + (time.time() - self.start) / 60
        )  # in minutes
        self.start = time.time()
        # a list with one value for each epoch
        self.stats["train_losses"] = self.train_losses 
        self.stats["train_accs"] = self.train_accs 

        self.stats["val_losses"] = self.val_losses 
        self.stats["val_accs"] = self.val_accs 

        self.stats["min_val_loss"] = self.min_val_loss
        self.stats["max_val_acc"] = self.max_val_acc
        self.stats["best_epoch"] = self.best_epoch
        torch.save(self.stats, self.stats_filename)

    def _checkpoint_model_and_opt(self, current_epoch: int):
        if self.dataparallel or self.gpu is not None:
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

    def _one_batch(self, batch: Tensor, labels: Tensor, backprop: bool = True):
        """
        Passes one batch of samples through model to get predictions & loss
        Does backprop if training 
        """
        self.model.zero_grad()
        preds = self.model(batch)                               # N x K x 3
        loss = torch.nn.BCEWithLogitsLoss(
                    reduction='sum'
                )(
                preds, labels
            )
        if backprop:            
            self.optimizer.zero_grad()
            loss.backward()

            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)

            self.optimizer.step()
        return loss.item(), preds.detach()
 
    def train(self):
        self.start = time.time()
        self.to_break = 0
        for epoch in range(self.begin_epoch, self.epochs + self.begin_epoch):
            self.model.train()
            train_loss, train_correct_preds = 0, 0
            epoch_train_size = 0
            train_loader = tqdm(self.train_loader, desc='training...')
            for i, batch in enumerate(train_loader):
                batch_data = batch[0].to(self.device)
                batch_labels = batch[1].to(self.device)
                batch_loss, batch_preds_logits = self._one_batch(
                    batch_data, batch_labels, backprop=True
                )

                train_loss += batch_loss
                train_batch_size = batch_preds_logits.shape[0]
                epoch_train_size += train_batch_size

                if self.lr_scheduler_name == 'CosineAnnealingWarmRestarts':
                    self.lr_scheduler.step(epoch + i / self.train_size - self.args.lr_scheduler_epoch_offset)
                elif self.lr_scheduler_name == 'OneCycleLR':
                    self.lr_scheduler.step()
                
                batch_preds = torch.sigmoid(batch_preds_logits)
                batch_correct_preds = torch.sum(torch.eq(batch_preds, batch_labels)).item() / 3 # bcos we have 3 models TODO: split into 3
                train_correct_preds += batch_correct_preds
                running_acc = train_correct_preds / epoch_train_size

                train_loader.set_description(f"training...loss={train_loss/epoch_train_size:.4f}, acc={running_acc:.4f}")
                train_loader.refresh()
            
            self.train_accs.append(train_correct_preds / epoch_train_size)
            self.train_losses.append(train_loss / epoch_train_size)

            # validation
            self.model.eval()
            with torch.no_grad(): 
                val_loss, val_correct_preds = 0, 0
                val_loader = tqdm(self.val_loader, desc='validating...')
                epoch_val_size = 0
                for i, batch in enumerate(val_loader):
                    batch_data = batch[0].to(self.device)
                    batch_labels = batch[1].to(self.device)
                    batch_loss, batch_preds_logits = self._one_batch(
                        batch_data, batch_labels, backprop=False
                    )
                    val_batch_size = batch_preds_logits.shape[0]
                    epoch_val_size += val_batch_size

                    batch_preds = torch.sigmoid(batch_preds_logits)
                    batch_correct_preds = torch.sum(torch.eq(batch_preds, batch_labels)).item() / 3 # bcos we have 3 models #TODO: split into 3
                    val_correct_preds += batch_correct_preds
                    running_acc = val_correct_preds / epoch_val_size
                    
                    if self.debug: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                        batch_idx = batch[2] # List
                        try:
                            for j in range(i * self.args.batch_size_eval, (i+1) * self.args.batch_size_eval):
                                if j % (self.val_size // 5) == random.randint(0, 3) or j % (self.val_size // 8) == random.randint(0, 4):  # peek at a random sample of current batch to monitor training progress
                                    prod_idx = random.sample(list(range(self.args.batch_size_eval)), k=1)[0]
                                    prod_smi = self.proposals_data['valid'][batch_idx[prod_idx], 0]
                                    prod_preds_logits = batch_preds_logits[prod_idx]
                                    prod_preds = batch_preds[prod_idx]
                                    prod_labels = batch_labels[prod_idx]
                                    logging.info(f'\nproduct SMILES:\t\t{prod_smi}')
                                    logging.info(f'GLN:\t\tlogits = {prod_preds_logits[0].item():+.4f}, hard = {prod_preds[0].item():.0f}, label = {prod_labels[0]:.0f}')
                                    logging.info(f'Retrosim:\t\tlogits = {prod_preds_logits[1].item():+.4f}, hard = {prod_preds[1].item():.0f}, label = {prod_labels[1]:.0f}')
                                    logging.info(f'RetroXpert:\t\tlogits = {prod_preds_logits[2].item():+.4f}, hard = {prod_preds[2].item():.0f}, label = {prod_labels[2]:.0f}')
                                    break
                        except Exception as e: # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                            logging.info("".join(tb_str))
                            logging.info('\nIndex out of range (last minibatch)')

                    val_loss += batch_loss
                    val_loader.set_description(f"validating...loss={val_loss/epoch_val_size:.4f}, acc={running_acc:.4f}")
                    val_loader.refresh()
                
                self.val_accs.append(val_correct_preds / epoch_val_size)
                self.val_losses.append(val_loss / epoch_val_size)

            # track best_epoch to facilitate loading of best checkpoint
            if self.early_stop_criteria == 'loss':
                if self.val_losses[-1] < self.min_val_loss:
                    self.best_epoch = epoch
                    self.min_val_loss = self.val_losses[-1]
                    is_best = True
            elif self.early_stop_criteria == 'acc':
                if self.val_accs[-1] > self.max_val_acc:
                    self.best_epoch = epoch
                    self.max_val_acc = self.val_accs[-1]
                    is_best = True
            
            if 'Feedforward' in self.model_name: # as FF-EBM weights are massive, only save if is_best
                if self.rank == 0 and self.checkpoint and is_best: # and (epoch - self.begin_epoch) % self.checkpoint_every == 0:
                    self._checkpoint_model_and_opt(current_epoch=epoch)
            else: # for G2E/S2E, models are small, is ok to save regularly
                if self.rank == 0 and self.checkpoint and (epoch - self.begin_epoch) % self.checkpoint_every == 0:
                    self._checkpoint_model_and_opt(current_epoch=epoch)

            self._update_stats()
            if self.early_stop:
                self._check_earlystop(current_epoch=epoch)
                if self.to_break:  # is it time to early stop?
                    break

            if self.lr_scheduler_name == 'ReduceLROnPlateau': # update lr scheduler if we are using one
                if self.args.lr_scheduler_criteria == 'loss':
                    self.lr_scheduler.step(self.val_losses[-1])
                elif self.args.lr_scheduler_criteria == 'acc': # monitor top-1 acc for lr_scheduler 
                    self.lr_scheduler.step(self.val_accs[-1])
                logging.info(f'\nCalled a step of ReduceLROnPlateau, current LR: {self.optimizer.param_groups[0]["lr"]}')

            message = f"\nEnd of epoch: {epoch}, \
                \ntrain loss: {self.train_losses[-1]:.4f}, train acc: {self.train_accs[-1]:.4f}, \
                \nval loss: {self.val_losses[-1]: .4f}, val acc: {self.val_accs[-1]:.4f}, \
                \n"
            logging.info(message)
            try:
                message += f'{self.expt_name}'
                send_message(message)
            except Exception as e:
                logging.info(e)
                logging.info("Don't worry about this - just a small hack to send messages to Telegram")
            if self.args.lr_floor_stop_training and self.optimizer.param_groups[0]['lr'] < self.args.lr_floor:
                logging.info('Stopping training as learning rate has dropped below 1e-6')
                break 

        logging.info(f'Total training time: {self.stats["train_time"]}')
        return False, None, None

    def test(self, saved_stats: Optional[dict] = None):
        self.model.eval()
        test_loss, test_correct_preds = 0, 0
        test_loader = tqdm(self.test_loader, desc='testing...')
        with torch.no_grad():
            epoch_test_size = 0
            for i, batch in enumerate(test_loader):
                batch_data = batch[0].to(self.device)
                batch_labels = batch[1].to(self.device)
                batch_loss, batch_preds_logits = self._one_batch(
                    batch_data, batch_labels, backprop=False
                )
                test_batch_size = batch_preds_logits.shape[0]
                epoch_test_size += test_batch_size

                batch_preds = torch.sigmoid(batch_preds_logits)
                batch_correct_preds = torch.sum(torch.eq(batch_preds, batch_labels)).item() / 3 # bcos we have 3 models TODO: split into 3
                test_correct_preds += batch_correct_preds
                running_acc = test_correct_preds / epoch_test_size

                if self.debug: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                    batch_idx = batch[2] # List
                    try:
                        for j in range(i * self.args.batch_size_eval, (i+1) * self.args.batch_size_eval):
                            if j % (self.test_size // 5) == random.randint(0, 3) or j % (self.test_size // 8) == random.randint(0, 4):  # peek at a random sample of current batch to monitor training progress
                                prod_idx = random.sample(list(range(self.args.batch_size_eval)), k=1)[0]
                                prod_smi = self.proposals_data['test'][batch_idx[prod_idx], 0]
                                prod_preds_logits = batch_preds_logits[prod_idx]
                                prod_preds = batch_preds[prod_idx]
                                prod_labels = batch_labels[prod_idx]
                                logging.info(f'\nproduct SMILES:\t\t{prod_smi}')
                                logging.info(f'GLN:\t\tlogits = {prod_preds_logits[0].item():+.4f}, hard = {prod_preds[0].item():.0f}, label = {prod_labels[0]:.0f}')
                                logging.info(f'Retrosim:\t\tlogits = {prod_preds_logits[1].item():+.4f}, hard = {prod_preds[1].item():.0f}, label = {prod_labels[1]:.0f}')
                                logging.info(f'RetroXpert:\t\tlogits = {prod_preds_logits[2].item():+.4f}, hard = {prod_preds[2].item():.0f}, label = {prod_labels[2]:.0f}')
                                break
                    except Exception as e: # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                        logging.info("".join(tb_str))
                        logging.info('\nIndex out of range (last minibatch)')

                test_loss += batch_loss
                test_loader.set_description(f"validating...loss={test_loss/epoch_test_size:.4f}, acc={running_acc:.4f}")
                test_loader.refresh()
                
        if saved_stats:
            self.stats = saved_stats
        if len(self.stats.keys()) <= 2:
            raise RuntimeError(
                "self.stats only has 2 keys or less. If loading checkpoint, you need to provide load_stats!"
            )

        self.stats["test_loss"] = test_loss / epoch_test_size # self.test_size 
        logging.info(f'\nTest loss: {self.stats["test_loss"]:.4f}')
        self.stats["test_acc"] = running_acc
        message = f"{self.expt_name}\n"
        this_topk_message = f'Test acc: {100 * self.stats["test_acc"]:.3f}%, \
                \ntest loss: {self.stats["test_loss"]:.4f} \
                \n'
        logging.info(this_topk_message)
        message += this_topk_message
        try:
            send_message(message)
        except Exception as e:
            pass
        torch.save(self.stats, self.stats_filename) # override existing train stats w/ train+test stats

    def train_distributed(self):
        raise NotImplementedError

    def test_distributed(self, saved_stats: Optional[dict] = None):
        raise NotImplementedError

    def get_preds_and_loss_distributed(
        self,
        phase: str = "test",
        save_preds: Optional[bool] = True,
        name_preds: Optional[Union[str, bytes, os.PathLike]] = None,
        path_to_preds: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> Tuple[Tensor, float, Tensor]:
        raise NotImplementedError('Please reload the best checkpoint and run expt.get_preds_and_loss() on a single GPU')

    def get_preds_and_loss(
        self,
        phase: str = "test",
        save_preds: Optional[bool] = True,
        name_preds: Optional[Union[str, bytes, os.PathLike]] = None,
        path_to_preds: Optional[Union[str, bytes, os.PathLike]] = None,
    ) -> Tuple[Tensor, float, Tensor]:
        if phase == "test":
            dataloader = self.test_loader
        elif phase == "train":
            dataloader = self.train_loader
        elif phase == "valid":
            dataloader = self.val_loader

        self.model.eval()
        with torch.no_grad():
            preds_combined, labels_combined = [], []
            epoch_data_size = 0
            total_loss = 0
            for batch in tqdm(dataloader, desc='getting raw logit outputs...'):
                batch_data = batch[0].to(self.device)
                batch_labels = batch[1].to(self.device)
                labels_combined.append(batch_labels)
                labels_combined = torch.cat(labels_combined, dim=0).squeeze(dim=-1).cpu()
                
                preds = self.model(batch_data)
                preds_combined.append(preds)
                preds_combined = torch.cat(preds_combined, dim=0).squeeze(dim=-1).cpu() 

                loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(
                    preds, batch_labels
                )
                epoch_data_size += preds.shape[0]
                total_loss += loss
            
            total_loss /= epoch_data_size
            logging.info(f"\nLoss on {phase} : {total_loss:.4f}")

        if path_to_preds is None:
            path_to_preds = Path(__file__).resolve().parents[1] / "preds"
        else:
            path_to_preds = Path(path_to_preds)
        if name_preds is None:
            name_preds = f"{self.model_name}_{self.expt_name}_preds_{phase}.pkl"
        if save_preds:
            logging.info(f"Saving raw logit predictions at: {Path(path_to_preds / name_preds)}")
            torch.save(preds_combined, Path(path_to_preds / name_preds))

        if phase == 'train':
            self.stats["train_loss_nodropout"] = total_loss
        self.preds[phase] = preds_combined
        return preds_combined, total_loss, labels_combined

    def get_acc(
        self,
        phase: str = "test",
    ):
        if phase not in self.preds:
            pred_logits, loss, labels = self.get_preds_and_loss(phase=phase)
            self.preds[phase] = pred_logits
            if phase == 'train':
                self.stats["train_loss_nodropout"] = loss
        
        pred_hard = torch.sigmoid(pred_logits)
        # bcos we have 3 models TODO: split into 3
        accuracy = (torch.sum(torch.eq(pred_hard, labels)).item() // 3) / labels.shape[0] * 100

        self.stats[f"{phase}_acc_nodropout"] = accuracy
        torch.save(self.stats, self.stats_filename)

        message = f"Accuracy ({phase}): {100 * accuracy:.3f}%"
        logging.info(message)
        message += f'\nLoss on {phase}: {loss:.4f}\n{self.expt_name}'
        try:
            send_message(message)
        except:
            pass
        return