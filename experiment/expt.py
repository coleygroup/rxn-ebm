from data.dataset import ReactionDataset, AugmentedData
from experiment.utils import _worker_init_fn_nmslib_, _worker_init_fn_default_
from model.utils import save_checkpoint, seed_everything
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import os
import numpy as np
from tqdm import tqdm
import time
import random
from typing import Optional, Union
from pathlib import Path
from collections import defaultdict

Tensor = torch.Tensor

'''
NOTE: the way python's imports work, is that I cannot run scripts on their own from that directory
(e.g. I cannot do python FF.py because FF.py imports functions from model.utils, and at the time of
executing FF.py, python has no knowledge of this package level information (i.e. __package__ is None))
therefore, I have to run/import all these files/functions from a script in the main directory,
i.e. same folder as trainEBM.py by doing: from model.FF import *; from experiment.expt import * etc

To run this script from terminal/interpreter, go to rxnebm/ then execute python -m experiment.expt
'''


class Experiment():
    '''
    NOTE: currently assumes pre-computed file already exists --> trainEBM.py handles the pre-computation
    TODO: wandb/tensorboard for hyperparameter tuning

    Parameters
    ----------
    device : Optional[str] (Default = None)
        'cuda' or 'cpu'
        device to do training, testing & inference on.
        If None, automatically detects if GPU is available, else uses CPU.
    load_checkpoint : Optional[bool] (Default = False)
        whether to load from a previous checkpoint.
        if True: load_optimizer, load_stats & begin_epoch must be provided
    '''

    def __init__(self,
                 model: nn.Module,
                 model_args: dict,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 early_stop: bool,
                 min_delta: float,
                 patience: int,
                 num_workers: int,
                 checkpoint: bool,
                 random_seed: int,
                 precomp_file_prefix: str,
                 checkpoint_folder: Union[str,
                                          bytes,
                                          os.PathLike],
                 expt_name: str,
                 rxn_type: str,
                 fp_type: str,
                 rctfp_size: int,
                 prodfp_size: int,
                 augmentations: dict,
                 onthefly: Optional[bool] = False,
                 lookup_dict_filename: Optional[Union[str,
                                                      bytes,
                                                      os.PathLike]] = None,
                 mol_fps_filename: Optional[Union[str,
                                                  bytes,
                                                  os.PathLike]] = None,
                 search_index_filename: Optional[Union[str,
                                                       bytes,
                                                       os.PathLike]] = None,
                 device: Optional[str] = None,
                 distributed: Optional[bool] = False,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 root: Optional[Union[str,
                                      bytes,
                                      os.PathLike]] = None,
                 load_checkpoint: Optional[bool] = False,
                 saved_optimizer: Optional[torch.optim.Optimizer] = None,
                 saved_stats: Optional[dict] = None,
                 begin_epoch: Optional[int] = None,
                 **kwargs):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stop = early_stop
        self.min_delta = min_delta
        self.patience = patience
        self.num_workers = num_workers
        self.checkpoint = checkpoint
        self.random_seed = random_seed

        self.rxn_type = rxn_type
        self.fp_type = fp_type
        self.rctfp_size = rctfp_size
        self.prodfp_size = prodfp_size
        self.fp_radius = kwargs['fp_radius']

        if root:
            self.root = Path(root)
        else:
            self.root = Path(__file__).parents[1] / 'data' / 'cleaned_data'
        self.checkpoint_folder = Path(checkpoint_folder)

        self.expt_name = expt_name
        self.augmentations = augmentations
        print('\nInitialising experiment: ', self.expt_name)
        print('Augmentations: ', self.augmentations)
        if device:
            self.device = device
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.model = model
        self.model_name = model.__repr__()
        self.distributed = distributed  # TODO: affects how checkpoint is saved
        self.best_epoch = 0  # will be automatically assigned after 1 epoch

        self._collate_args(model_args, optimizer, saved_optimizer)

        if load_checkpoint:
            self._load_checkpoint(saved_optimizer, saved_stats, begin_epoch)
        else:
            self._init_opt_and_stats(optimizer)

        # if onthefly is True, need lookup_dict_filename, mol_fps_filename, (if
        # doing cosine: search_index_filename)
        self._init_dataloaders(
            precomp_file_prefix,
            onthefly,
            lookup_dict_filename,
            mol_fps_filename,
            search_index_filename)
        seed_everything(random_seed)

    def __repr__(self):
        return 'Experiment with: ' + self.augmentations

    def _collate_args(self, model_args: dict,
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      saved_optimizer: Optional[torch.optim.Optimizer] = None):
        self.model_args = model_args
        self.fp_args = {
            'rxn_type': self.rxn_type,
            'fp_type': self.fp_type,
            'rctfp_size': self.rctfp_size,
            'prodfp_size': self.prodfp_size,
            'fp_radius': self.fp_radius
        }
        self.train_args = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'early_stop': self.early_stop,
            'min_delta': self.min_delta,
            'patience': self.patience,
            'num_workers': self.num_workers,
            'checkpoint': self.checkpoint,
            'random_seed': self.random_seed,
            'expt_name': self.expt_name,
            'device': self.device,
            'model_name': self.model_name,
            'distributed': self.distributed,
            # prioritise optimizer over saved_optimizer
            'optimizer': optimizer or saved_optimizer,
        }

    def _load_checkpoint(
            self,
            saved_optimizer: torch.optim.Optimizer,
            saved_stats: dict,
            begin_epoch: int):
        print('Loading checkpoint...')

        if saved_optimizer is None:
            raise ValueError('load_checkpoint requires saved_optimizer!')
        self.optimizer = saved_optimizer  # load optimizer w/ state dict from checkpoint

        if saved_stats is None:
            raise ValueError('load_checkpoint requires saved_stats!')
        self.stats = saved_stats
        self.stats_filename = self.checkpoint_folder / \
            f'{self.model_name}_{self.expt_name}_stats.pkl'

        self.model_args = self.stats['model_args']
        self.fp_args = self.stats['fp_args']
        self.train_args = self.stats['train_args']
        self.augmentations = self.stats['augmentations']

        self.train_losses = self.stats['train_losses']
        self.train_accs = self.stats['train_accs']
        self.val_losses = self.stats['val_losses']
        self.val_accs = self.stats['val_accs']
        self.min_val_loss = self.stats['min_val_loss']
        self.wait = 0  # counter for _check_earlystop()

        if begin_epoch is None:
            raise ValueError('load_checkpoint requires begin_epoch!')
        self.begin_epoch = begin_epoch

    def _init_opt_and_stats(self, optimizer: torch.optim.Optimizer):
        # to store training statistics
        print('Initialising optimizer & stats...')
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=self.learning_rate)

        self.train_losses = []
        self.train_accs = []
        self.min_val_loss = float('+inf')
        self.val_losses = []
        self.val_accs = []
        self.begin_epoch = 0
        self.wait = 0  # counter for _check_earlystop()

        self.stats = {
            'model_args': self.model_args,
            'fp_args': self.fp_args,
            'train_args': self.train_args,
            'augmentations': self.augmentations,
            'train_time': 0
        }
        self.stats_filename = self.checkpoint_folder / \
            f'{self.model_name}_{self.expt_name}_stats.pkl'

    def _init_dataloaders(
            self,
            precomp_file_prefix: str,
            onthefly: bool,
            lookup_dict_filename: str,
            mol_fps_filename: str,
            search_index_filename: str,
            rxn_smis_file_prefix: Optional[str] = None):
        print('Initialising dataloaders...')
        if self.num_workers > 0:
            if 'cos' in self.augmentations.keys() or 'cosine' in self.augmentations.keys():
                worker_init_fn = _worker_init_fn_nmslib_
            else:
                worker_init_fn = _worker_init_fn_default_
        else:  # i.e. num_workers == 0
            worker_init_fn = None

        rxn_smis_filenames = defaultdict(str)
        precomp_filenames = defaultdict(str)
        if onthefly:
            augmented_data = AugmentedData(
                self.augmentations,
                lookup_dict_filename,
                mol_fps_filename,
                search_index_filename)
            for dataset in [
                'train',
                'valid',
                    'test']:  # rxn_smis_file_prefix = '50k_clean_rxnsmi_noreagent'
                rxn_smis_filenames[dataset] = rxn_smis_file_prefix + \
                    f'_{dataset}.pickle'
        else:
            augmented_data = None
            for dataset in [
                'train',
                'valid',
                    'test']:  # precomp_file_prefix = '50k_count_rdm_5'
                precomp_filenames[dataset] = precomp_file_prefix + \
                    f'_{dataset}.npz'

        if self.rxn_type == 'sep':
            input_dim = self.rctfp_size + self.prodfp_size
        elif self.rxn_type == 'diff':
            input_dim = self.rctfp_size

        pin_memory = True if torch.cuda.is_available() else False
        train_dataset = ReactionDataset(
            input_dim,
            precomp_filenames['train'],
            rxn_smis_filenames['train'],
            onthefly,
            augmented_data)
        self.train_loader = DataLoader(
            train_dataset, self.batch_size,
            num_workers=self.num_workers, worker_init_fn=worker_init_fn,
            shuffle=True, pin_memory=pin_memory)
        self.train_size = len(train_dataset)

        val_dataset = ReactionDataset(
            input_dim,
            precomp_filenames['valid'],
            rxn_smis_filenames['valid'],
            onthefly,
            augmented_data)
        self.val_loader = DataLoader(
            val_dataset, self.batch_size,
            num_workers=self.num_workers, worker_init_fn=worker_init_fn,
            shuffle=False, pin_memory=pin_memory)
        self.val_size = len(val_dataset)

        test_dataset = ReactionDataset(
            input_dim,
            precomp_filenames['test'],
            rxn_smis_filenames['test'],
            onthefly,
            augmented_data)
        self.test_loader = DataLoader(
            test_dataset, self.batch_size,
            num_workers=self.num_workers, worker_init_fn=worker_init_fn,
            shuffle=False, pin_memory=pin_memory)
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset  # save memory

    def _check_earlystop(self, current_epoch):
        self.to_break = 0
        if self.min_val_loss - self.val_losses[-1] < self.min_delta:
            if self.patience <= self.wait:
                print(f'\nEarly stopped at the end of epoch: {current_epoch}, \
                train loss: {self.train_losses[-1]:.4f}, top-1 train acc: {self.train_accs[-1]:.4f}, \
                \nval loss: {self.val_losses[-1]:.4f}, top-1 val acc: {self.val_accs[-1]:.4f}')
                self.stats['early_stop_epoch'] = current_epoch
                self.to_break = 1  # will break loop
            else:
                self.wait += 1
                print(
                    'Decrease in val loss < min_delta, patience count: ',
                    self.wait)
        else:
            self.wait = 0
            self.min_val_loss = min(self.min_val_loss, self.val_losses[-1])

    def _update_stats(self):
        self.stats['train_time'] = self.stats['train_time'] + \
            (time.time() - self.start) / 60  # in minutes
        self.start = time.time()

        # a list with one value for each epoch
        self.stats['train_losses'] = self.train_losses
        # a list with one value for each epoch
        self.stats['train_accs'] = self.train_accs
        # a list with one value for each epoch
        self.stats['val_losses'] = self.val_losses
        # a list with one value for each epoch
        self.stats['val_accs'] = self.val_accs
        self.stats['min_val_loss'] = self.min_val_loss
        self.stats['best_epoch'] = self.best_epoch
        torch.save(self.stats, self.stats_filename)

    def _checkpoint_model_and_opt(self, current_epoch: int):
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        checkpoint_dict = {
            'epoch': current_epoch,  # epochs are 0-indexed
            'model_name': self.model_name,
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats
        }
        checkpoint_filename = self.checkpoint_folder / \
            f'{self.model_name}_{self.expt_name}_checkpoint_{current_epoch:04d}.pth.tar'
        torch.save(checkpoint_dict, checkpoint_filename)

    def _one_batch(self, batch: Tensor, backprop: bool = True):
        '''
        Passes one batch of samples through model to get scores & loss
        Does backprop if training
        TODO: learning rate scheduler
        '''
        for p in self.model.parameters():
            p.grad = None  # faster, equivalent to self.model.zero_grad()
        scores = self.model(batch)  # size N x K

        # positives are the 0-th index of each sample
        loss = (scores[:, 0] + torch.logsumexp(-scores, dim=1)).sum()

        if backprop:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # index with lowest energy is the most feasible rxn
        pred_labels = torch.topk(scores, 1, dim=1, largest=False)[1]
        # 0-th index should have the lowest energy
        pred_correct = torch.where(pred_labels == 0)[0].shape[0]
        return loss.item(), pred_correct

    def train(self):
        self.start = time.time()  # timeit.default_timer()
        for epoch in range(self.begin_epoch, self.epochs):
            self.model.train()
            train_loss, train_correct_preds = 0, 0
            for batch in tqdm(self.train_loader):
                batch = batch.to(self.device)
                curr_batch_loss, curr_batch_correct_preds = self._one_batch(
                    batch, backprop=True)
                train_loss += curr_batch_loss
                train_correct_preds += curr_batch_correct_preds
            self.train_accs.append(train_correct_preds / self.train_size)
            self.train_losses.append(train_loss / self.train_size)

            self.model.eval()
            val_loss, val_correct_preds = 0, 0
            for batch in tqdm(self.val_loader):
                batch = batch.to(self.device)
                curr_batch_loss, curr_batch_correct_preds = self._one_batch(
                    batch, backprop=False)
                val_loss += curr_batch_loss
                val_correct_preds += curr_batch_correct_preds
            self.val_accs.append(val_correct_preds / self.val_size)
            self.val_losses.append(val_loss / self.val_size)
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

            print(f'\nEnd of epoch: {epoch}, \
                \ntrain loss: {self.train_losses[-1]:.6f}, top-1 train acc: {self.train_accs[-1]:.4f}, \
                \nval loss: {self.val_losses[-1]: .6f}, top-1 val acc: {self.val_accs[-1]:.4f}')

        print(f'Total training time: {self.stats["train_time"]}')

    def test(self, saved_stats: Optional[dict] = None):
        '''
        Evaluates the model on the test set
        Parameters
        ---------
        saved_stats: Optional[dict]
            Test statistics will be stored inside this stats file
            Used to load existing stats file when loading a trained model from checkpoint
        '''
        self.model.eval()
        test_loss, test_correct_preds = 0, 0
        for batch in tqdm(self.test_loader):
            batch = batch.to(self.device)
            curr_batch_loss, curr_batch_correct_preds = self._one_batch(
                batch, backprop=False)
            test_loss += curr_batch_loss
            test_correct_preds += curr_batch_correct_preds

        if saved_stats:
            self.stats = saved_stats
        if len(self.stats.keys()) <= 2:
            raise RuntimeError(
                'self.stats only has 2 keys or less. If loading checkpoint, you need to provide load_stats!')

        self.stats['test_loss'] = test_loss / self.test_size
        self.stats['test_acc'] = test_correct_preds / self.test_size
        print(f'Test loss: {self.stats["test_loss"]}')
        print(f'Test top-1 accuracy: {self.stats["test_acc"]}')
        # overrides existing training stats w/ training + test stats
        torch.save(self.stats, self.stats_filename)

    def get_scores_and_loss(self,
                            dataset_name: Optional[str] = 'test',
                            dataloader: Optional[torch.utils.data.DataLoader] = None,
                            dataset_len: Optional[int] = None,
                            show_neg: Optional[bool] = False) -> Tensor:
        '''
        Gets raw energy values (scores) from a trained model on a given dataloader,
        with the option to save pos_neg_smis to analyse model performance

        Parameters
        ----------
        dataset_name : str (Default = 'test')
            choose from 'train', 'test', or 'valid'
            whether to get scores from train/test/valid datasets
        dataloader : Optional[Dataloader] (Default = None)
            custom dataloader that loops through dataset that is not the original train, test or val
            (which are simply accessed by providing a key parameter)
        dataset_len : Optional[int] (Default = None)
            length of dataset used for dataloader

        Returns
        -------
        scores : Tensor
            scores of shape (# rxns, 1 + # neg rxns)
        loss : float
            the loss value on the provided dataset

        TODO: fix show_neg: index into SMILES molecule vocab to retrieve molecules -->
        save as groups [true product/rct SMILES, 1st NN SMILES, ... K-1'th NN SMILES])
        '''
        if dataloader is not None:
            if dataset_len is None:
                raise ValueError(
                    'Please provide the size of the custom dataset')
            dataset_name = 'custom'
        elif dataset_name == 'test':
            dataloader = self.test_loader
        elif dataset_name == 'train':
            dataloader = self.train_loader
        elif dataset_name == 'val':
            dataloader == self.val_loader

        self.model.eval()
        scores = []
        with torch.no_grad():
            if show_neg:  # save neg rxn_smis to analyse model performance
                print('Showing negative examples is not functional now! Sorry!')
                return
#                 pos_neg_smis = []
#                 for pos_neg_smi, batch in tqdm(dataloader):
#                     batch = batch.to(self.device)
#                     scores.append(self.model(batch).cpu()) # scores: size N x K
#                     pos_neg_smis.append(pos_neg_smi)
#                     del batch

#                 torch.save(pos_neg_smis, self.trainargs['checkpoint_path']+'{}_{}_posnegsmi.pkl'.format(
# self.trainargs['model'], self.trainargs['expt_name']))
            else:
                for batch in tqdm(dataloader):
                    batch = batch.to(self.device)
                    scores.append(self.model(batch).cpu())

            scores = torch.cat(scores, dim=0).squeeze(dim=-1)
            loss = (scores[:, 0] + torch.logsumexp(-1 * scores, dim=1)).sum()
            if dataset_len is not None:  # means using a custom dataset other than train/valid/test
                loss /= dataset_len
            elif dataset_name == 'test':
                loss /= self.test_size
            elif dataset_name == 'train':
                loss /= self.train_size
            elif dataset_name == 'val':
                loss /= self.val_size
            print(f'Loss on {dataset_name} : {loss.item():.6f}')
#             if show_neg:
#                 return scores, pos_neg_smis
#             else:
            return scores, loss
            # output shape N x K: where N = # positive rxns in dataset;
            # K = 1 + # negative rxns per positive rxn

    def get_topk_acc(self,
                     dataset_name: Optional[str] = 'test',
                     save_train: Optional[bool] = True,
                     dataloader: Optional[torch.utils.data.DataLoader] = None,
                     dataset_len: Optional[int] = None,
                     k: Optional[int] = 1,
                     repeats: Optional[int] = 1,
                     save_scores: Optional[bool] = True,
                     name_scores: Optional[Union[str,
                                                 bytes,
                                                 os.PathLike]] = None,
                     path_scores: Optional[Union[str,
                                                 bytes,
                                                 os.PathLike]] = None) -> Tensor:
        '''
        Computes top-k accuracy of trained model in classifying feasible vs infeasible chemical rxns
        (i.e. minimum energy assigned to label 0 of each training sample)

        Parameters
        ----------
        save_train : Optional[bool] (Default = True)
            whether to save train acc & loss without dropout into stats dictionary
        save_scores: bool (Default = True)
            whether to save the generated scores tensor
        name_scores: Union[str, bytes, os.PathLike] (Default = None)
            filename of scores to save as a .pkl file
            If None, automatically set to 'scores_<dataset_name>_<self.expt_name>'

        Returns
        -------
        scores: tensor
            scores of shape (# rxns, 1 + # neg rxns)

        Also see: self.get_scores_and_loss()
        '''
        accs = np.array([])
        for repeat in range(repeats):
            print('Running repeat: ', repeat)
            scores, loss = self.get_scores_and_loss(
                dataset_name=dataset_name, dataloader=dataloader, dataset_len=dataset_len)
            pred_labels = torch.topk(scores, k, dim=1, largest=False)[1]
            accs = np.append(accs, torch.where(pred_labels == 0)[
                             0].shape[0] / pred_labels.shape[0])

        if path_scores is None:
            path_scores = Path(__file__).parents[1] / 'scores'
        if name_scores is None:
            name_scores = f'scores_{dataset_name}_{self.expt_name}.pkl'
        if save_scores:
            print('Saving scores at: ', Path(path_scores / name_scores))
            torch.save(scores, Path(path_scores / name_scores))
        if dataset_name == 'train' and save_train:
            self.stats['train_loss_nodropout'] = loss
            self.stats['train_acc_nodropout'] = accs
            torch.save(self.stats, self.stats_filename)

        print('Top-1 accuracies: ', accs)
        print('Avg top-1 accuracy: ', accs.mean())
        print('Variance: ', accs.var())
        return scores
