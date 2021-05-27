import argparse
import logging
import math
import os
import random
import time
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from rxnebm.data import dataset, dataset_utils
from rxnebm.experiment import expt_utils
from rxnebm.model import model_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Tensor = torch.Tensor
try:
    send_message = partial(expt_utils.send_message, 
                       chat_id=os.environ['CHAT_ID'], 
                       bot_token=os.environ['BOT_TOKEN'])
except Exception as e:
    pass

ReLU = nn.ReLU()

class Experiment:
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        model_name: str,
        gpu: Optional[str] = None,
        root: Optional[Union[str, bytes, os.PathLike]] = None,
        load_checkpoint: Optional[bool] = False,
        saved_optimizer: Optional[torch.optim.Optimizer] = None,
        vocab: Dict[str, int] = None
    ):
        logging.info(f"\nInitialising experiment: {args.expt_name}")
        self.args = args
        self.vocab = vocab

        if self.args.early_stop:
            self.min_val_loss = float("+inf")  
            self.max_val_acc = float("-inf") 
            self.wait = 0  # counter for _check_earlystop()
        else:
            self.min_val_loss = float("+inf") # still needed to track best_epoch
            self.max_val_acc = float("-inf") # still needed to track best_epoch
            self.wait = None

        if root:
            self.root = Path(root) 
            logging.info(f'Using data root as user input {self.root}')
        else:
            self.root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"
            logging.info(f'Using default data root of {self.root}')

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

        if saved_optimizer is not None:
            self.optimizer_name = str(self.args.optimizer).split(' ')[0]
        else:
            self.optimizer_name = args.optimizer
        self.lr_scheduler_name = args.lr_scheduler

        self.proposals_data = {}
        self.proposals_csv_filenames = defaultdict(str)
        for phase in ['train', 'valid', 'test']:
            self.proposals_csv_filenames[phase] = self.args.proposals_csv_file_prefix + f"_{phase}.csv"
            if phase != 'train': # this is just for visualization purpose during val/test
                self.proposals_data[phase] = pd.read_csv(self.root / self.proposals_csv_filenames[phase], index_col=None, dtype='str')
                self.proposals_data[phase] = self.proposals_data[phase].drop(['orig_rxn_smi'], axis=1).values

        if self.args.representation == 'fingerprint':
            self._init_fp_dataloaders(precomp_rxnfp_prefix=args.precomp_rxnfp_prefix)
            self.maxk = self.train_loader.dataset.data.shape[-1] // self.train_loader.dataset.input_dim
        elif self.args.representation in ["smiles", "graph"]:
            self._init_smi_dataloaders()
            self.maxk = len(self.train_loader.dataset._rxn_smiles_with_negatives[0])
        else:
            raise NotImplementedError('Please add self.maxk for the current representation; '
                                      'we need maxk to calculate topk_accs up to the maximum allowed value '
                                      '(otherwise torch will raise index out of range)')
            # e.g. we can't calculate top-50 acc if we only have 1 positive + 10 negatives per minibatch

        if load_checkpoint:
            self._load_checkpoint(saved_optimizer)
        else:
            self._init_optimizer_and_stats()

        self.best_epoch = 0  # will be tracked during training
        self.train_losses = []
        self.val_losses = []
        self.train_topk_accs = defaultdict(lambda: [])
        self.val_topk_accs = defaultdict(lambda: [])
        self.test_topk_accs = defaultdict(lambda: np.nan)

        self.k_to_calc = [1, 2, 3, 5, 10, 20, 50]
        for i in reversed(range(len(self.k_to_calc))):
            # reversed so that removing that k won't affect current i wrt future k
            if self.k_to_calc[i] > self.maxk:
                self.k_to_calc.pop(i)

        self.k_to_test = [1, 2, 3, 5, 10, 20, 50]
        for i in reversed(range(len(self.k_to_test))):
            # reversed so that removing that k won't affect current i wrt future k
            if self.k_to_test[i] > self.args.minibatch_eval:
                self.k_to_test.pop(i)
        
        self.energies = {} # for self.get_topk_acc
        self.true_ranks = {} # for self.get_topk_acc

    def __repr__(self):
        return f"Experiment name: {self.args.expt_name}"

    def _load_checkpoint(
        self,
        saved_optimizer: torch.optim.Optimizer,
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
        else:
            self.lr_scheduler = None

    def _init_optimizer_and_stats(self): 
        logging.info("Initialising optimizer & stats...")
        self.optimizer = model_utils.get_optimizer(self.optimizer_name)(self.model.parameters(), lr=self.args.learning_rate)
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
        else:
            logging.info('Not using any LR Scheduler!')
            self.lr_scheduler = None

    def _init_fp_dataloaders(
        self,
        precomp_rxnfp_prefix: str,
    ):
        logging.info("Initialising dataloaders...")
        precomp_rxnfp_filenames = defaultdict(str) # precomputed rxn fp files (retro proposals)
        for phase in ["train", "valid", "test"]:
            precomp_rxnfp_filenames[phase] = precomp_rxnfp_prefix + f"_{phase}.npz"

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
            root=self.root
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
            self.args.batch_size,
            shuffle=True if self.train_sampler is None else False,
            sampler=self.train_sampler
        )
        self.train_size = len(train_dataset)

        val_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_rxnfp_filenames["valid"],
            root=self.root
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
            shuffle=False,
            sampler=self.val_sampler
        )
        self.val_size = len(val_dataset)

        test_dataset = dataset.ReactionDatasetFingerprints(
            input_dim=input_dim,
            precomp_rxnfp_filename=precomp_rxnfp_filenames["test"],
            root=self.root
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
            shuffle=False,
            sampler=self.test_sampler
        )
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset

    def _get_smi_dl(self, phase: str, shuffle: bool = False):        
        _data = dataset.ReactionDatasetSMILES(
            self.args,
            phase=phase,
            proposals_csv_filename=self.proposals_csv_filenames[phase],
            root=self.root
        )

        if self.args.representation == 'graph':
            collate_fn = dataset_utils.graph_collate_fn_builder(
                self.device, debug=False)
        else:
            collate_fn = dataset_utils.seq_collate_fn_builder(
                self.device, self.vocab, self.args.max_seq_len, debug=False)
        
        if self.gpu is not None:
            _sampler = torch.utils.data.distributed.DistributedSampler(
                _data,
                num_replicas=self.args.world_size,
                rank=self.rank
            )
            shuffle = False
        else:
            _sampler = None

        _loader = DataLoader(
            _data,
            batch_size=self.args.batch_size if phase == 'train' else self.args.batch_size_eval,
            shuffle=shuffle,
            collate_fn=collate_fn,
            sampler=_sampler
        )

        _size = len(_data)
        del _data

        return _loader, _size, _sampler

    def _init_smi_dataloaders(self):
        logging.info("Initialising SMILES/Graph dataloaders...")
        self.train_loader, self.train_size, self.train_sampler = self._get_smi_dl(phase="train", shuffle=True)
        self.val_loader, self.val_size, self.val_sampler = self._get_smi_dl(phase="valid", shuffle=False)
        self.test_loader, self.test_size, self.test_sampler = self._get_smi_dl(phase="test", shuffle=False)

    def _check_earlystop(self, current_epoch):
        if self.args.early_stop_criteria == 'loss': 
            if self.min_val_loss - self.val_losses[-1] < 0:
                if self.args.early_stop_patience <= self.wait:
                    logging.info(
                        f"\nEarly stopped at the end of epoch: {current_epoch}, \
                    train loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_topk_accs[1][-1]:.4f}, \
                    \nval loss: {self.val_losses[-1]:.4f}, top-1 val acc: {self.val_topk_accs[1][-1]:.4f}"
                    )
                    self.to_break = 1  # will break loop
                else:
                    self.wait += 1
                    logging.info(f"\nDecrease in val loss < 0, \
                                patience count: {self.wait}")
            else:
                self.wait = 0
                self.min_val_loss = min(self.min_val_loss, self.val_losses[-1])

        elif self.args.early_stop_criteria.split('_')[-1] == 'acc':
            k = int(self.args.early_stop_criteria.split('_')[0][-1:])
            val_acc_to_compare = self.val_topk_accs[k][-1] 

            if self.max_val_acc - val_acc_to_compare > 0:
                if self.args.early_stop_patience <= self.wait:
                    message = f"\nEarly stopped at the end of epoch: {current_epoch}, \
                    \ntrain loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_topk_accs[1][-1]:.4f}, \
                    \nval loss: {self.val_losses[-1]:.4f}, top-1 val acc: {self.val_topk_accs[1][-1]:.4f} \
                    \n"
                    logging.info(message)
                    if self.rank == 0 or self.rank is None:
                        try:
                            message += f'{self.args.expt_name}'
                            send_message(message)
                        except Exception as e:
                            pass
                    self.to_break = 1  # will break loop
                else:
                    self.wait += 1
                    logging.info(
                        f'\nIncrease in top-{k} val acc < 0, \
                        \npatience count: {self.wait} \
                        \n')
            else:
                self.wait = 0
                self.max_val_acc = max(self.max_val_acc, val_acc_to_compare)

    def _checkpoint_model_and_opt(self, current_epoch: int):
        if self.gpu is not None:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        checkpoint_dict = {
            "epoch": current_epoch,  # epochs are 0-indexed
            "model_name": self.model_name,
            "state_dict": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
        }
        checkpoint_filename = (
            self.args.checkpoint_folder
            / f"{self.model_name}_{self.args.expt_name}_checkpoint.pth.tar"
        )
        torch.save(checkpoint_dict, checkpoint_filename)

    def _one_batch(
        self, batch: Tensor, mask: Tensor, backprop: bool = True):
        """
        Passes one batch of samples through model to get energies & loss
        Does backprop if training 
        """
        self.model.zero_grad()
        energies = self.model(batch)  # size N x K
        # replace all-zero vectors with float('inf'), making those gradients 0 on backprop
        energies = torch.where(mask, energies, torch.tensor([float('inf')], device=mask.device))
        if backprop:
            # for training only: positives are the 0-th index of each minibatch (row)
            loss = (energies[:, 0] + torch.logsumexp(-energies, dim=1)).sum()
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)

            self.optimizer.step()
            return loss.item(), energies.cpu().detach()
        else:
            # for validation/testing, cannot calculate loss naively
            # need the true_ranks of precursors from retrosynthesis models
            return energies.cpu().detach()

    def train(self):
        self.start = time.time()
        self.to_break = 0
        running_topk_accs = defaultdict(lambda: np.nan)

        for epoch in range(self.args.begin_epoch, self.args.epochs + self.args.begin_epoch):
            self.model.train()
            train_loss, train_correct_preds = 0, defaultdict(int)
            epoch_train_size = 0
            train_loader = tqdm(self.train_loader, desc='training...')
            for i, batch in enumerate(train_loader):
                batch_data = batch[0]
                if not isinstance(batch_data, tuple):
                    batch_data = batch_data.to(self.device)
                if self.model_name == 'TransformerEBM':
                    batch_data = (batch_data, 'train')
                batch_mask = batch[1].to(self.device)
                batch_loss, batch_energies = self._one_batch(
                    batch_data, batch_mask, backprop=True,
                )
                train_loss += batch_loss

                if math.isnan(train_loss):
                    msg = 'Training loss is nan and training diverged!'
                    logging.info(msg)
                    msg += f'\n{self.args.expt_name}'
                    try:
                        send_message(message)
                    except Exception as e:
                        logging.info(e)
                    raise ValueError('Training loss is nan')
                    
                train_batch_size = batch_energies.shape[0]
                epoch_train_size += train_batch_size
                    
                for k in self.k_to_calc: # various top-k accuracies
                    # index with lowest energy is what the model deems to be the most feasible rxn 
                    batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]
                    # for training, true rxn is always at 0th-index 
                    batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0]
                    train_correct_preds[k] += batch_correct_preds
                    running_topk_accs[k] = train_correct_preds[k] / epoch_train_size
                    
                train_loader.set_description(f"training...loss={train_loss/epoch_train_size:.4f}, top-1 acc={running_topk_accs[1]:.4f}, top-5 acc={running_topk_accs[5]:.4f}, top-10 acc={running_topk_accs[10]:.4f}")
                train_loader.refresh()
            
            for k in self.k_to_calc:
                self.train_topk_accs[k].append(train_correct_preds[k] / epoch_train_size) 
            self.train_losses.append(train_loss / epoch_train_size)

            # validation
            self.model.eval()
            with torch.no_grad():
                val_loss, val_correct_preds = 0, defaultdict(int)
                val_loader = tqdm(self.val_loader, desc='validating...')
                epoch_val_size = 0
                for i, batch in enumerate(val_loader):
                    batch_data = batch[0]
                    if not isinstance(batch_data, tuple):
                        batch_data = batch_data.to(self.device)
                    if self.model_name == 'TransformerEBM':
                        batch_data = (batch_data, 'valid')
                    batch_mask = batch[1].to(self.device)
                    batch_energies = self._one_batch(
                        batch_data, batch_mask, backprop=False
                    )
                    val_batch_size = batch_energies.shape[0]
                    epoch_val_size += val_batch_size

                    # for validation/test data, true rxn may not be present!
                    batch_idx = batch[2] # List
                    batch_true_ranks_array = self.proposals_data['valid'][batch_idx, 2].astype('int')
                    batch_true_ranks_valid = batch_true_ranks_array[batch_true_ranks_array < self.args.minibatch_eval]
                    batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                    # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation
                    # (bcos nothing in the numerator, loss is undefined)
                    loss_numerator = batch_energies[
                        np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                        batch_true_ranks_valid
                    ]
                    loss_denominator = batch_energies[
                        np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                        :
                    ]
                    batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item()
                    for k in self.k_to_calc:
                        # index with lowest energy is what the model deems to be the most feasible rxn
                        batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]
                        batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                        val_correct_preds[k] += batch_correct_preds
                        running_topk_accs[k] = val_correct_preds[k] / epoch_val_size

                        if k == 1:
                            # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                            try:
                                for j in range(i * self.args.batch_size_eval, (i+1) * self.args.batch_size_eval):
                                    if j % (self.val_size // 5) == random.randint(0, 1) or j % (self.val_size // 8) == random.randint(0, 2):  # peek at a random sample of current batch to monitor training progress
                                        rxn_idx = random.sample(list(range(self.args.batch_size_eval)), k=1)[0]
                                        rxn_true_rank = batch_true_ranks_array[rxn_idx]
                                        rxn_pred_rank = batch_preds[rxn_idx, 0].item()
                                        rxn_pred_energy = batch_energies[rxn_idx, rxn_pred_rank].item()
                                        rxn_true_energy = batch_energies[rxn_idx, rxn_true_rank].item() if rxn_true_rank != 9999 else 'NaN'
                                        rxn_orig_energy = batch_energies[rxn_idx, 0].item()
                                        rxn_orig_energy2 = batch_energies[rxn_idx, 1].item()
                                        rxn_orig_energy3 = batch_energies[rxn_idx, 2].item()

                                        rxn_true_prod = self.proposals_data['valid'][batch_idx[rxn_idx], 0]
                                        rxn_true_prec = self.proposals_data['valid'][batch_idx[rxn_idx], 1]
                                        rxn_cand_precs = self.proposals_data['valid'][batch_idx[rxn_idx], 3:]
                                        rxn_pred_prec = rxn_cand_precs[batch_preds[rxn_idx]]
                                        rxn_orig_prec = rxn_cand_precs[0]
                                        rxn_orig_prec2 = rxn_cand_precs[1]
                                        rxn_orig_prec3 = rxn_cand_precs[2]
                                        logging.info(f'\ntrue product:                          \t\t\t\t{rxn_true_prod}')
                                        logging.info(f'pred precursor (rank {rxn_pred_rank}, energy = {rxn_pred_energy:+.4f}):\t\t\t{rxn_pred_prec}')
                                        if rxn_true_energy == 'NaN':
                                            logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy}):\t\t\t\t{rxn_true_prec}')
                                        else:
                                            logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy:+.4f}):\t\t\t{rxn_true_prec}')
                                        logging.info(f'orig precursor (rank 0, energy = {rxn_orig_energy:+.4f}):\t\t\t{rxn_orig_prec}')
                                        logging.info(f'orig precursor (rank 1, energy = {rxn_orig_energy2:+.4f}):\t\t\t{rxn_orig_prec2}')
                                        logging.info(f'orig precursor (rank 2, energy = {rxn_orig_energy3:+.4f}):\t\t\t{rxn_orig_prec3}\n')
                                        break
                            except Exception as e: # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                                tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                                logging.info("".join(tb_str))
                                logging.info('\nIndex out of range (last minibatch)')

                    val_loss += batch_loss
                    val_loader.set_description(f"validating...loss={val_loss/epoch_val_size:.4f}, top-1 acc={running_topk_accs[1]:.4f}, top-5 acc={running_topk_accs[5]:.4f}, top-10 acc={running_topk_accs[10]:.4f}")
                    val_loader.refresh()
                
                for k in self.k_to_calc:
                    self.val_topk_accs[k].append(val_correct_preds[k] / epoch_val_size)
                self.val_losses.append(val_loss / epoch_val_size)

            is_best = False
            # track best_epoch to facilitate loading of best checkpoint
            if self.args.early_stop_criteria == 'loss':
                if self.val_losses[-1] < self.min_val_loss:
                    self.best_epoch = epoch
                    self.min_val_loss = self.val_losses[-1]
                    is_best = True
            elif self.args.early_stop_criteria.split('_')[-1] == 'acc':
                k = int(self.args.early_stop_criteria.split('_')[0][-1:])
                val_acc_to_compare = self.val_topk_accs[k][-1]
                if val_acc_to_compare > self.max_val_acc:
                    self.best_epoch = epoch
                    self.max_val_acc = val_acc_to_compare
                    is_best = True
            
            if (self.rank is None or self.rank == 0) and self.args.checkpoint and is_best:
                self._checkpoint_model_and_opt(current_epoch=epoch)
            if self.args.early_stop:
                self._check_earlystop(current_epoch=epoch)
                if self.to_break:  # is it time to early stop?
                    break

            if self.lr_scheduler_name == 'ReduceLROnPlateau': # update lr scheduler if we are using one 
                if self.args.lr_scheduler_criteria == 'loss':
                    self.lr_scheduler.step(self.val_losses[-1])
                elif self.args.lr_scheduler_criteria == 'acc': # monitor top-1 acc for lr_scheduler 
                    self.lr_scheduler.step(self.val_topk_accs[1][-1])
                logging.info(f'\nCalled a step of ReduceLROnPlateau, current LR: {self.optimizer.param_groups[0]["lr"]}')

            epoch_train_accs = defaultdict(lambda: np.nan)
            epoch_val_accs = defaultdict(lambda: np.nan)
            for k in self.k_to_calc:
                epoch_train_accs[k] = self.train_topk_accs[k][-1]
            for k in self.k_to_test:
                epoch_val_accs[k] = self.val_topk_accs[k][-1]
                
            message = f"\nEnd of epoch: {epoch}, \
                \ntrain loss: {self.train_losses[-1]:.4f}, top-1 train acc: {epoch_train_accs[1]:.4f}, \
                \ntop-3 train acc: {epoch_train_accs[3]:.4f}, top-5 train acc: {epoch_train_accs[5]:.4f}, \
                \ntop-10 train acc: {epoch_train_accs[10]:.4f}, top-20 train acc: {epoch_train_accs[20]:.4f}, \
                \ntop-50 train acc: {epoch_train_accs[50]:.4f}, \
                \nval loss: {self.val_losses[-1]: .4f}, top-1 val acc: {epoch_val_accs[1]:.4f}, \
                \ntop-3 val acc: {epoch_val_accs[3]:.4f}, top-5 val acc: {epoch_val_accs[5]:.4f} \
                \ntop-10 val acc: {epoch_val_accs[10]:.4f}, top-20 train acc: {epoch_val_accs[20]:.4f}, \
                \ntop-50 train acc: {epoch_val_accs[50]:.4f} \
                \n"
            logging.info(message)
            try:
                message += f'{self.args.expt_name}'
                send_message(message)
            except Exception as e:
                logging.info(e)
                logging.info("Don't worry about this - just a small hack to send messages to Telegram")
            if self.args.lr_floor_stop_training and self.optimizer.param_groups[0]['lr'] < self.args.lr_floor:
                logging.info('Stopping training as learning rate has dropped below 1e-6')
                break

        logging.info(f'Total training time: {(time.time() - self.start) / 60:.2f} mins')

    def train_distributed(self):
        torch.cuda.synchronize()
        self.start = time.time()
        self.to_break = 0
        self.val_sampler.set_epoch(0) # no need to change this throughout training
        running_topk_accs = defaultdict(lambda: np.nan)

        for epoch in range(self.args.begin_epoch, self.args.epochs + self.args.begin_epoch):
            self.train_sampler.set_epoch(epoch)
            self.model.train()

            train_loss, train_correct_preds = 0, defaultdict(int)
            epoch_train_size = 0
            if self.rank == 0: # or self.rank is None:
                train_loader = tqdm(self.train_loader, desc='training...')
            else:
                train_loader = self.train_loader

            for i, batch in enumerate(train_loader):
                batch_data = batch[0]
                if not isinstance(batch_data, tuple):
                    batch_data = batch_data.cuda(non_blocking=True)
                if self.model_name == 'TransformerEBM':
                    batch_data = (batch_data, 'train')
                batch_mask = batch[1].cuda(non_blocking=True)
                batch_loss, batch_energies = self._one_batch(
                    batch_data, batch_mask, backprop=True,
                )
                batch_loss = torch.tensor([batch_loss]).cuda(self.gpu, non_blocking=True)
                dist.all_reduce(batch_loss, dist.ReduceOp.SUM)
                batch_loss = batch_loss.item()
                train_loss += batch_loss

                if math.isnan(train_loss):
                    if self.rank == 0:
                        msg = 'Training loss is nan and training diverged!'
                        logging.info(msg)
                        msg += f'\n{self.args.expt_name}'
                        try:
                            send_message(message)
                        except Exception as e:
                            logging.info(e)
                    raise ValueError('Training loss is nan')

                train_batch_size = batch_energies.shape[0]
                train_batch_size = torch.tensor([train_batch_size]).cuda(self.gpu, non_blocking=True)
                dist.all_reduce(train_batch_size, dist.ReduceOp.SUM)
                train_batch_size = train_batch_size.item()
                epoch_train_size += train_batch_size
                    
                for k in self.k_to_calc: # various top-k accuracies
                    # index with lowest energy is what the model deems to be the most feasible rxn 
                    batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]
                    # for training, true rxn is always at 0th-index 
                    batch_correct_preds = torch.where(batch_preds == 0)[0].shape[0]
                    batch_correct_preds = torch.tensor([batch_correct_preds]).cuda(self.gpu, non_blocking=True)
                    dist.all_reduce(batch_correct_preds, dist.ReduceOp.SUM)
                    batch_correct_preds = batch_correct_preds.item()
                    train_correct_preds[k] += batch_correct_preds                    
                    running_topk_accs[k] = train_correct_preds[k] / epoch_train_size
                
                if self.rank == 0:
                    train_loader.set_description(f"training (epoch {epoch}): loss={train_loss/epoch_train_size:.4f}, top-1 acc={running_topk_accs[1]:.4f}, top-5 acc={running_topk_accs[5]:.4f}, top-10 acc={running_topk_accs[10]:.4f}")
                    train_loader.refresh()
            
            for k in self.k_to_calc:
                self.train_topk_accs[k].append(train_correct_preds[k] / epoch_train_size)
            self.train_losses.append(train_loss / epoch_train_size)

            # validation
            self.model.eval()
            with torch.no_grad():
                val_loss, val_correct_preds = 0, defaultdict(int)
                if self.rank == 0:
                    val_loader = tqdm(self.val_loader, desc='validating...')
                else:
                    val_loader = self.val_loader
                epoch_val_size = 0
                for i, batch in enumerate(val_loader):
                    batch_data = batch[0]
                    if not isinstance(batch_data, tuple):
                        batch_data = batch_data.cuda(non_blocking=True)
                    if self.model_name == 'TransformerEBM':
                        batch_data = (batch_data, 'valid')
                    batch_mask = batch[1].cuda(non_blocking=True)
                    batch_energies = self._one_batch(
                        batch_data, batch_mask, backprop=False
                    )
                    val_batch_size = batch_energies.shape[0]
                    val_batch_size = torch.tensor([val_batch_size]).cuda(self.gpu, non_blocking=True)
                    dist.all_reduce(val_batch_size, dist.ReduceOp.SUM)
                    val_batch_size = val_batch_size.item()
                    epoch_val_size += val_batch_size

                    # for validation/test data, true rxn may not be present!
                    batch_idx = batch[2] # List
                    batch_true_ranks_array = self.proposals_data['valid'][batch_idx, 2].astype('int')
                    batch_true_ranks_valid = batch_true_ranks_array[batch_true_ranks_array < self.args.minibatch_eval]
                    batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                    # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation
                    # (bcos nothing in the numerator, loss is undefined)
                    loss_numerator = batch_energies[
                        np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                        batch_true_ranks_valid
                    ]
                    loss_denominator = batch_energies[
                        np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                        :
                    ]
                    batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item()

                    for k in self.k_to_test:
                        # index with lowest energy is what the model deems to be the most feasible rxn
                        batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]
                        batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                        batch_correct_preds = torch.tensor([batch_correct_preds]).cuda(self.gpu, non_blocking=True)
                        dist.all_reduce(batch_correct_preds, dist.ReduceOp.SUM)
                        batch_correct_preds = batch_correct_preds.item()
                        val_correct_preds[k] += batch_correct_preds
                        running_topk_accs[k] = val_correct_preds[k] / epoch_val_size

                        if k == 1 and self.rank == 0: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                            try:
                                for j in range(i * self.args.batch_size_eval, (i+1) * self.args.batch_size_eval):
                                    if j % (self.val_size // 5) == random.randint(0, 3) or j % (self.val_size // 8) == random.randint(0, 4):  # peek at a random sample of current batch to monitor training progress
                                        rxn_idx = random.sample(list(range(self.args.batch_size_eval)), k=1)[0]
                                        rxn_true_rank = batch_true_ranks_array[rxn_idx]
                                        rxn_pred_rank = batch_preds[rxn_idx, 0].item()
                                        rxn_pred_energy = batch_energies[rxn_idx, rxn_pred_rank].item()
                                        rxn_true_energy = batch_energies[rxn_idx, rxn_true_rank].item() if rxn_true_rank != 9999 else 'NaN'
                                        rxn_orig_energy = batch_energies[rxn_idx, 0].item()
                                        rxn_orig_energy2 = batch_energies[rxn_idx, 1].item()
                                        rxn_orig_energy3 = batch_energies[rxn_idx, 2].item()

                                        rxn_true_prod = self.proposals_data['valid'][batch_idx[rxn_idx], 0]
                                        rxn_true_prec = self.proposals_data['valid'][batch_idx[rxn_idx], 1]
                                        rxn_cand_precs = self.proposals_data['valid'][batch_idx[rxn_idx], 3:]
                                        rxn_pred_prec = rxn_cand_precs[batch_preds[rxn_idx]]
                                        rxn_orig_prec = rxn_cand_precs[0]
                                        rxn_orig_prec2 = rxn_cand_precs[1]
                                        rxn_orig_prec3 = rxn_cand_precs[2]
                                        logging.info(f'\ntrue product:                          \t\t\t\t{rxn_true_prod}')
                                        logging.info(f'pred precursor (rank {rxn_pred_rank}, energy = {rxn_pred_energy:+.4f}):\t\t\t{rxn_pred_prec}')
                                        if rxn_true_energy == 'NaN':
                                            logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy}):\t\t\t\t{rxn_true_prec}')
                                        else:
                                            logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy:+.4f}):\t\t\t{rxn_true_prec}')
                                        logging.info(f'orig precursor (rank 0, energy = {rxn_orig_energy:+.4f}):\t\t\t{rxn_orig_prec}')
                                        logging.info(f'orig precursor (rank 1, energy = {rxn_orig_energy2:+.4f}):\t\t\t{rxn_orig_prec2}')
                                        logging.info(f'orig precursor (rank 2, energy = {rxn_orig_energy3:+.4f}):\t\t\t{rxn_orig_prec3}\n')
                                        break
                            except Exception as e: # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                                tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                                logging.info("".join(tb_str))
                                logging.info('\nIndex out of range (last minibatch)')

                    batch_loss = torch.tensor([batch_loss]).cuda(self.gpu, non_blocking=True)
                    dist.all_reduce(batch_loss, dist.ReduceOp.SUM)
                    batch_loss = batch_loss.item()
                    val_loss += batch_loss

                    if self.rank == 0:
                        val_loader.set_description(f"validating (epoch {epoch}): loss={val_loss/epoch_val_size:.4f}, top-1 acc={running_topk_accs[1]:.4f}, top-5 acc={running_topk_accs[5]:.4f}, top-10 acc={running_topk_accs[10]:.4f}")
                        val_loader.refresh()
 
                for k in self.k_to_test:
                    self.val_topk_accs[k].append(val_correct_preds[k] / epoch_val_size)
                self.val_losses.append(val_loss / epoch_val_size)

            is_best = False
            # track best_epoch to facilitate loading of best checkpoint
            if self.args.early_stop_criteria == 'loss':
                if self.val_losses[-1] < self.min_val_loss:
                    self.best_epoch = epoch
                    self.min_val_loss = self.val_losses[-1]
                    is_best = True
            elif self.args.early_stop_criteria.split('_')[-1] == 'acc':
                k = int(self.args.early_stop_criteria.split('_')[0][-1:])
                val_acc_to_compare = self.val_topk_accs[k][-1]
                if val_acc_to_compare > self.max_val_acc:
                    self.best_epoch = epoch
                    self.max_val_acc = val_acc_to_compare
                    is_best = True
            
            if self.rank == 0 and self.args.checkpoint and is_best:
                self._checkpoint_model_and_opt(current_epoch=epoch)

            dist.barrier() # wait for process 0 to save model checkpoint
            torch.cuda.synchronize()
            if self.args.early_stop:
                self._check_earlystop(current_epoch=epoch)
                if self.to_break:  # is it time to early stop?
                    break

            if self.lr_scheduler_name == 'ReduceLROnPlateau': # update lr scheduler if we are using one 
                if self.args.lr_scheduler_criteria == 'loss':
                    self.lr_scheduler.step(self.val_losses[-1])
                elif self.args.lr_scheduler_criteria == 'acc': # monitor top-1 acc for lr_scheduler 
                    self.lr_scheduler.step(self.val_topk_accs[1][-1])
                logging.info(f'\nCalled a step of ReduceLROnPlateau, current LR: {self.optimizer.param_groups[0]["lr"]}')

            epoch_train_accs = defaultdict(lambda: np.nan)
            epoch_val_accs = defaultdict(lambda: np.nan)
            for k in self.k_to_calc:
                epoch_train_accs[k] = self.train_topk_accs[k][-1]
            for k in self.k_to_test:
                epoch_val_accs[k] = self.val_topk_accs[k][-1]

            if self.rank == 0:
                message = f"\nEnd of epoch: {epoch}, \
                    \ntrain loss: {self.train_losses[-1]:.4f}, top-1 train acc: {epoch_train_accs[1]:.4f}, \
                    \ntop-3 train acc: {epoch_train_accs[3]:.4f}, top-5 train acc: {epoch_train_accs[5]:.4f}, \
                    \ntop-10 train acc: {epoch_train_accs[10]:.4f}, top-20 train acc: {epoch_train_accs[20]:.4f}, \
                    \ntop-50 train acc: {epoch_train_accs[50]:.4f}, \
                    \nval loss: {self.val_losses[-1]: .4f}, top-1 val acc: {epoch_val_accs[1]:.4f}, \
                    \ntop-3 val acc: {epoch_val_accs[3]:.4f}, top-5 val acc: {epoch_val_accs[5]:.4f} \
                    \ntop-10 val acc: {epoch_val_accs[10]:.4f}, top-20 train acc: {epoch_val_accs[20]:.4f}, \
                    \ntop-50 train acc: {epoch_val_accs[50]:.4f} \
                    \n"
                logging.info(message)
                try:
                    message += f'{self.args.expt_name}'
                    send_message(message)
                except Exception as e:
                    logging.info(e)
                    logging.info("Don't worry about this - just a small hack to send messages to Telegram")
                if self.args.lr_floor_stop_training and self.optimizer.param_groups[0]['lr'] < self.args.lr_floor:
                    logging.info('Stopping training as learning rate has dropped below 1e-6')
                    break

        if self.rank == 0:
            logging.info(f'Total training time: {(time.time() - self.start) / 60:.2f} mins')

    def test(self):
        """
        Evaluates the model on the test set
        """
        self.model.eval()
        test_loss, test_correct_preds = 0, defaultdict(int)
        if self.test_loader is None: # running G2E
            self.test_loader, self.test_size, _ = self._get_smi_dl(phase="test", shuffle=False)
        test_loader = tqdm(self.test_loader, desc='testing...')

        running_topk_accs = defaultdict(lambda: np.nan)
        with torch.no_grad():
            epoch_test_size = 0
            for i, batch in enumerate(test_loader):
                batch_data = batch[0]
                if not isinstance(batch_data, tuple):
                    batch_data = batch_data.to(self.device)
                if self.model_name == 'TransformerEBM':
                    batch_data = (batch_data, 'test')
                batch_mask = batch[1].to(self.device)
                batch_energies = self._one_batch(
                    batch_data, batch_mask, backprop=False,
                )
                test_batch_size = batch_energies.shape[0]
                epoch_test_size += test_batch_size

                # for validation/test data, true rxn may not be present!
                batch_idx = batch[2]
                batch_true_ranks_array = self.proposals_data['test'][batch_idx, 2].astype('int')
                batch_true_ranks_valid = batch_true_ranks_array[batch_true_ranks_array < self.args.minibatch_eval]
                batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation
                # (bcos nothing in the numerator, loss is undefined)
                loss_numerator = batch_energies[
                    np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                    batch_true_ranks_valid
                ]
                loss_denominator = batch_energies[
                    np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                    :
                ]
                batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item()
                for k in self.k_to_test:
                    # index with lowest energy is what the model deems to be the most feasible rxn
                    batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]  
                    batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                    test_correct_preds[k] += batch_correct_preds
                    running_topk_accs[k] = test_correct_preds[k] / epoch_test_size

                    if k == 1:
                        # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                        try:
                            for j in range(i * self.args.batch_size_eval, (i+1) * self.args.batch_size_eval):
                                if j % (self.test_size // 5) == random.randint(0, 3) or j % (self.test_size // 8) == random.randint(0, 5):  # peek at a random sample of current batch to monitor training progress
                                    rxn_idx = random.sample(list(range(self.args.batch_size_eval)), k=1)[0]
                                    rxn_true_rank = batch_true_ranks_array[rxn_idx]
                                    rxn_pred_rank = batch_preds[rxn_idx, 0].item()
                                    rxn_pred_energy = batch_energies[rxn_idx, rxn_pred_rank].item()
                                    rxn_true_energy = batch_energies[rxn_idx, rxn_true_rank].item() if rxn_true_rank != 9999 else 'NaN'
                                    rxn_orig_energy = batch_energies[rxn_idx, 0].item()
                                    rxn_orig_energy2 = batch_energies[rxn_idx, 1].item()
                                    rxn_orig_energy3 = batch_energies[rxn_idx, 2].item()

                                    rxn_true_prod = self.proposals_data['test'][batch_idx[rxn_idx], 0]
                                    rxn_true_prec = self.proposals_data['test'][batch_idx[rxn_idx], 1]
                                    rxn_cand_precs = self.proposals_data['test'][batch_idx[rxn_idx], 3:]
                                    rxn_pred_prec = rxn_cand_precs[batch_preds[rxn_idx]]
                                    rxn_orig_prec = rxn_cand_precs[0]
                                    rxn_orig_prec2 = rxn_cand_precs[1]
                                    rxn_orig_prec3 = rxn_cand_precs[2]
                                    logging.info(f'\ntrue product:                          \t\t\t\t{rxn_true_prod}')
                                    logging.info(f'pred precursor (rank {rxn_pred_rank}, energy = {rxn_pred_energy:+.4f}):\t\t\t{rxn_pred_prec}')
                                    if rxn_true_energy == 'NaN':
                                        logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy}):\t\t\t\t{rxn_true_prec}')
                                    else:
                                        logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy:+.4f}):\t\t\t{rxn_true_prec}')
                                    logging.info(f'orig precursor (rank 0, energy = {rxn_orig_energy:+.4f}):\t\t\t{rxn_orig_prec}')
                                    logging.info(f'orig precursor (rank 1, energy = {rxn_orig_energy2:+.4f}):\t\t\t{rxn_orig_prec2}')
                                    logging.info(f'orig precursor (rank 2, energy = {rxn_orig_energy3:+.4f}):\t\t\t{rxn_orig_prec3}\n')
                                    break
                        except Exception as e:
                            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                            logging.info("".join(tb_str))
                            logging.info('\nIndex out of range (last minibatch)')

                test_loss += batch_loss
                test_loader.set_description(f"testing...loss={test_loss / epoch_test_size:.4f}, top-1 acc={running_topk_accs[1]:.4f}, top-5 acc={running_topk_accs[5]:.4f}, top-10 acc={running_topk_accs[10]:.4f}")
                test_loader.refresh()
                
            for k in self.k_to_test:
                self.test_topk_accs[k] = test_correct_preds[k] / epoch_test_size # self.test_size

        logging.info(f'\nTest loss: {test_loss / epoch_test_size:.4f}')
        message = f"{self.args.expt_name}\n"
        for k in self.k_to_test:
            this_topk_message = f'Test top-{k} accuracy: {100 * self.test_topk_accs[k]:.3f}%'
            logging.info(this_topk_message)
            message += this_topk_message + '\n'
        try:
            send_message(message)
        except Exception as e:
            pass

    def test_distributed(self):
        """
        Evaluates the model on the test set
        """
        self.model.eval()
        test_loss, test_correct_preds = 0, defaultdict(int)
        if self.test_loader is None: # running G2E
            self.test_loader, self.test_size, self.test_sampler = self._get_smi_dl(phase="test", shuffle=False)
        self.test_sampler.set_epoch(0)
        if self.rank == 0:
            test_loader = tqdm(self.test_loader, desc='testing...')
        else:
            test_loader = self.test_loader
        
        running_topk_accs = defaultdict(lambda: np.nan)
        with torch.no_grad():
            epoch_test_size = 0
            for i, batch in enumerate(test_loader):
                batch_data = batch[0]
                if not isinstance(batch_data, tuple):
                    batch_data = batch_data.cuda(non_blocking=True)
                if self.model_name == 'TransformerEBM':
                    batch_data = (batch_data, 'test')
                batch_mask = batch[1].cuda(non_blocking=True)
                batch_energies = self._one_batch(
                    batch_data, batch_mask, backprop=False,
                )
                test_batch_size = batch_energies.shape[0]
                test_batch_size = torch.tensor([test_batch_size]).cuda(self.gpu, non_blocking=True)
                dist.all_reduce(test_batch_size, dist.ReduceOp.SUM)
                test_batch_size = test_batch_size.item()
                epoch_test_size += test_batch_size

                # for validation/test data, true rxn may not be present!
                batch_idx = batch[2]
                batch_true_ranks_array = self.proposals_data['test'][batch_idx, 2].astype('int')
                batch_true_ranks_valid = batch_true_ranks_array[batch_true_ranks_array < self.args.minibatch_eval]
                batch_true_ranks = torch.as_tensor(batch_true_ranks_array).unsqueeze(dim=-1)
                # slightly tricky as we have to ignore rxns with no 'positive' rxn for loss calculation
                # (bcos nothing in the numerator, loss is undefined)
                loss_numerator = batch_energies[
                    np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                    batch_true_ranks_valid
                ]
                loss_denominator = batch_energies[
                    np.arange(batch_energies.shape[0])[batch_true_ranks_array < self.args.minibatch_eval],
                    :
                ]
                batch_loss = (loss_numerator + torch.logsumexp(-loss_denominator, dim=1)).sum().item()

                for k in self.k_to_test:
                    # index with lowest energy is what the model deems to be the most feasible rxn
                    batch_preds = torch.topk(batch_energies, k=k, dim=1, largest=False)[1]  
                    batch_correct_preds = torch.where(batch_preds == batch_true_ranks)[0].shape[0]
                    batch_correct_preds = torch.tensor([batch_correct_preds]).cuda(self.gpu, non_blocking=True)
                    dist.all_reduce(batch_correct_preds, dist.ReduceOp.SUM)
                    batch_correct_preds = batch_correct_preds.item()
                    test_correct_preds[k] += batch_correct_preds
                    running_topk_accs[k] = test_correct_preds[k] / epoch_test_size

                    if k == 1 and self.rank == 0: # overhead is only 5 ms, will check ~5 times each epoch (regardless of batch_size)
                        try:
                            for j in range(i * self.args.batch_size_eval, (i+1) * self.args.batch_size_eval):
                                if j % (self.test_size // 5) == random.randint(0, 3) or j % (self.test_size // 8) == random.randint(0, 5):  # peek at a random sample of current batch to monitor training progress
                                    rxn_idx = random.sample(list(range(self.args.batch_size_eval)), k=1)[0]
                                    rxn_true_rank = batch_true_ranks_array[rxn_idx]
                                    rxn_pred_rank = batch_preds[rxn_idx, 0].item()
                                    rxn_pred_energy = batch_energies[rxn_idx, rxn_pred_rank].item()
                                    rxn_true_energy = batch_energies[rxn_idx, rxn_true_rank].item() if rxn_true_rank != 9999 else 'NaN'
                                    rxn_orig_energy = batch_energies[rxn_idx, 0].item()
                                    rxn_orig_energy2 = batch_energies[rxn_idx, 1].item()
                                    rxn_orig_energy3 = batch_energies[rxn_idx, 2].item()

                                    rxn_true_prod = self.proposals_data['test'][batch_idx[rxn_idx], 0]
                                    rxn_true_prec = self.proposals_data['test'][batch_idx[rxn_idx], 1]
                                    rxn_cand_precs = self.proposals_data['test'][batch_idx[rxn_idx], 3:]
                                    rxn_pred_prec = rxn_cand_precs[batch_preds[rxn_idx]]
                                    rxn_orig_prec = rxn_cand_precs[0]
                                    rxn_orig_prec2 = rxn_cand_precs[1]
                                    rxn_orig_prec3 = rxn_cand_precs[2]
                                    logging.info(f'\ntrue product:                          \t\t\t\t{rxn_true_prod}')
                                    logging.info(f'pred precursor (rank {rxn_pred_rank}, energy = {rxn_pred_energy:+.4f}):\t\t\t{rxn_pred_prec}')
                                    if rxn_true_energy == 'NaN':
                                        logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy}):\t\t\t\t{rxn_true_prec}')
                                    else:
                                        logging.info(f'true precursor (rank {rxn_true_rank}, energy = {rxn_true_energy:+.4f}):\t\t\t{rxn_true_prec}')
                                    logging.info(f'orig precursor (rank 0, energy = {rxn_orig_energy:+.4f}):\t\t\t{rxn_orig_prec}')
                                    logging.info(f'orig precursor (rank 1, energy = {rxn_orig_energy2:+.4f}):\t\t\t{rxn_orig_prec2}')
                                    logging.info(f'orig precursor (rank 2, energy = {rxn_orig_energy3:+.4f}):\t\t\t{rxn_orig_prec3}\n')
                                    break
                        except Exception as e:
                            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                            logging.info("".join(tb_str))
                            logging.info('\nIndex out of range (last minibatch)')
                
                batch_loss = torch.tensor([batch_loss]).cuda(self.gpu, non_blocking=True)
                dist.all_reduce(batch_loss, dist.ReduceOp.SUM)
                batch_loss = batch_loss.item()
                test_loss += batch_loss
                if self.rank == 0:
                    test_loader.set_description(f"testing...loss={test_loss / test_batch_size:.4f}, top-1 acc={running_topk_accs[1]:.4f}, top-5 acc={running_topk_accs[5]:.4f}, top-10 acc={running_topk_accs[10]:.4f}")
                    test_loader.refresh()
                
            for k in self.k_to_test:
                self.test_topk_accs[k] = test_correct_preds[k] / epoch_test_size
        
        dist.barrier()
        message = f"{self.args.expt_name}\n"
        if self.rank == 0:
            logging.info(f'\nTest loss: {test_loss / epoch_test_size:.4f}')
            for k in self.k_to_test:
                this_topk_message = f'Test top-{k} accuracy: {100 * self.test_topk_accs[k]:.3f}%'
                logging.info(this_topk_message)
                message += this_topk_message + '\n'
            try:
                send_message(message)
            except Exception as e:
                pass

    def get_energies_and_loss(
        self,
        phase: str = "test",
        save_energies: Optional[bool] = False,
    ) -> Tuple[Tensor, float]:
        """
        Gets raw energy values from a trained model on a given dataloader
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
            if phase != 'train':
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
                        batch_data, batch_mask, backprop=False,
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
            else: # training data, the positive (published) reaction is always at the 0-th index
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
                loss /= epoch_data_size
            logging.info(f"\nLoss on {phase} : {loss:.4f}")

        if save_energies:
            name_energies = f"{self.model_name}_{self.args.expt_name}_energies_{phase}.pkl"
            path_to_energies = Path(__file__).resolve().parents[1] / "energies"
            logging.info(f"Saving energies at: {Path(path_to_energies / name_energies)}")
            torch.save(energies_combined, Path(path_to_energies / name_energies))

        self.energies[phase] = energies_combined
        if phase != 'train':
            self.true_ranks[phase] = true_ranks.unsqueeze(dim=-1)
            return energies_combined, loss, true_ranks
        else:
            return energies_combined, loss

    def get_topk_acc(
        self,
        phase: str = "test",
        k: int = 1,
        message: Optional[str] = "",
    ) -> Tensor:
        """
        Returns
        -------
        energies: tensor
            energies of shape (# rxns, 1 + # neg rxns)

        Also see: self.get_energies_and_loss()
        """
        if phase != 'train':
            if phase not in self.energies:
                energies, loss, true_ranks = self.get_energies_and_loss(phase=phase)
            if self.energies[phase].shape[1] >= k:
                pred_labels = torch.topk(self.energies[phase], k=k, dim=1, largest=False)[1]
                topk_accuracy = torch.where(pred_labels == self.true_ranks[phase])[0].shape[0] / pred_labels.shape[0]

                this_topk_message = f"Top-{k} acc ({phase}): {100 * topk_accuracy:.3f}%"
                logging.info(this_topk_message)
                message += this_topk_message + '\n'
            else:
                logging.info(f'{k} out of range for dimension 1 on ({phase})')

        else: # true rank is always 0
            if phase not in self.energies:
                energies, loss = self.get_energies_and_loss(phase=phase)
                self.energies[phase] = energies
            
            if self.energies[phase].shape[1] >= k:
                pred_labels = torch.topk(self.energies[phase], k=k, dim=1, largest=False)[1]
                topk_accuracy = torch.where(pred_labels == 0)[0].shape[0] / pred_labels.shape[0]

                this_topk_message = f"Top-{k} acc ({phase}): {100 * topk_accuracy:.3f}%"
                logging.info(this_topk_message)
                message += this_topk_message + '\n'
            else:
                logging.info(f'{k} out of range for dimension 1 on ({phase})')
        return message
