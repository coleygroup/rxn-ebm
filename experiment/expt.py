import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm
import time
import random
from typing import Optional
from pathlib import Path
from collections import defaultdict

''' 
NOTE: the way python's imports work, is that I cannot run scripts on their own from that directory
(e.g. I cannot do python FF.py because FF.py imports functions from model.utils, and at the time of 
executing FF.py, python has no knowledge of this package level information (i.e. __package__ is None))
therefore, I have to run/import all these files/functions from a script in the main directory, 
i.e. same folder as trainEBM.py by doing: from model.FF import *; from experiment.expt import * and so on.
Then, it works as intended.
'''

from model.utils import save_checkpoint, seed_everything
from experiment.utils import _worker_init_fn_nmslib_, _worker_init_fn_default_
from data.dataset import ReactionDataset
from data.augmentors import Augmentor

class Experiment():
    '''
    NOTE: currently assumes pre-computed file already exists --> trainEBM.py handles the pre-computation
    TODO: finish onthefly augmentation support in self._init_dataloaders() 
    TODO: assumes that pre-requisite SMILES files already exist, so user just has to supply the path
    to those SMILES files, lookup_dict, count_mol_fps --> should have been handled in trainEBM.py or earlier 
    TODO: wandb/tensorboard for hyperparameter tuning
    TODO: fix batchloss --> just keep adding += , then divide by len(dataset) at the end of each epoch 

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
    def __init__(self, model: nn.Module, batch_size: int, learning_rate: float, epochs: int,
                early_stop: bool, min_delta: float, patience: int, num_workers: int, checkpoint: bool,
                random_seed: int, precomp_file_prefix: str, checkpoint_folder: str, expt_name: str, 
                rxn_type: str, fp_type: str, rctfp_size: int, prodfp_size: int,
                augmentations: dict, onthefly: Optional[bool]=False,  
                lookup_dict_filename: Optional[str]=None, mol_fps_filename: Optional[str]=None,
                search_index_filename: Optional[str]=None, device: Optional[str]=None, distributed: Optional[bool]=False,
                optimizer: Optional[torch.optim.Optimizer]=None, root: Optional[str]=None,
                saved_optimizer: Optional[torch.optim.Optimizer]=None, load_checkpoint: Optional[bool]=False, 
                saved_stats: Optional[dict]=None, saved_stats_filename: Optional[str]=None, begin_epoch: Optional[int]=None,
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

        if root:
            self.root = root
        else:
            self.root = Path(__file__).parents[1] / 'data' / 'cleaned_data'
        self.checkpoint_folder = checkpoint_folder

        self.expt_name = expt_name
        self.augmentations = augmentations
        print('Initialising experiment: ', self.expt_name)
        print('Augmentations: ', self.augmentations)
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.model = model
        self.model_name = model.__repr__()
        self.distributed = distributed # TODO: affects how checkpoint is saved
        self.best_epoch = None # will be automatically assigned after 1 epoch

        self._collate_args()

        if load_checkpoint: 
            self._load_checkpoint(saved_optimizer, saved_stats, saved_stats_filename, begin_epoch)
        else:  
            self._init_opt_and_stats(optimizer)

        self._init_dataloaders(precomp_file_prefix, onthefly, lookup_dict_filename, mol_fps_filename, search_index_filename) 
        seed_everything(random_seed)

    def __repr__(self):
        return 'Experiment with: ' + augmentations

    def _collate_args(self):
        self.args = {
            'batch_size' : self.batch_size, 
            'learning_rate' : self.learning_rate,
            'epochs' : self.epochs,
            'early_stop' : self.early_stop,
            'min_delta' : self.min_delta,
            'patience' : self.patience,
            'num_workers' : self.num_workers,
            'checkpoint' : self.checkpoint,
            'random_seed' : self.random_seed,
            'expt_name' : self.expt_name,
            'device' : self.device, 
            'model_name' : self.model_name, 
            'augmentations': self.augmentations,
            'distributed' : self.distributed, 
        }
    
    def _load_checkpoint(self, saved_optimizer: torch.optim.Optimizer, saved_stats: dict, 
                        saved_stats_filename: str, begin_epoch: int):
        print('Loading checkpoint...')
        assert saved_optimizer is not None, 'load_checkpoint requires saved_optimizer!'
        self.optimizer = saved_optimizer # load optimizer w/ state dict from checkpoint

        assert saved_stats is not None, 'load_checkpoint requires saved_stats!'
        assert saved_stats_filename is not None, 'load_checkpoint requires saved_stats_filename!'
        self.stats = saved_stats
        self.stats_filename = self.checkpoint_folder + saved_stats_filename
        self.stats['args'] = self.args
        self.mean_train_loss = self.stats['mean_train_loss']
        self.mean_train_acc = self.stats['mean_train_acc']
        self.mean_val_loss = self.stats['mean_val_loss']
        self.mean_val_acc = self.stats['mean_val_acc']
        self.min_val_loss = self.stats['min_val_loss']
        self.wait = 0 # counter for _check_earlystop()
        
        assert begin_epoch is not None, 'load_checkpoint requires begin_epoch!'
        self.begin_epoch = begin_epoch
    
    def _init_opt_and_stats(self, optimizer: torch.optim.Optimizer):
        print('Initialising optimizer & stats...') # to store training statistics  
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.mean_train_loss = []
        self.mean_train_acc = []
        self.min_val_loss = float('+inf')
        self.mean_val_loss = []
        self.mean_val_acc = []
        self.begin_epoch = 0
        self.wait = 0 # counter for _check_earlystop()
        self.stats = {'args': self.args, 'train_time': 0} 
        self.stats_filename = self.checkpoint_folder + f'{self.model_name}_{self.expt_name}_stats.pkl' 
    
    def _init_dataloaders(self, precomp_file_prefix: str, onthefly: bool, lookup_dict_filename: str, mol_fps_filename: str,
                        search_index_filename: str, rxn_smis_file_prefix: Optional[str]=None):
        print('Initialising dataloaders...')
        if self.num_workers > 0:
            if 'cos' in self.augmentations.keys() or 'cosine' in self.augmentations.keys():
                worker_init_fn = _worker_init_fn_nmslib_
            else:
                worker_init_fn = _worker_init_fn_default_
        else: # i.e. num_workers == 0
            worker_init_fn = None

        rxn_smis_filenames = defaultdict(str)
        precomp_filenames = defaultdict(str)
        if onthefly:
            augmentor = Augmentor(
                self.augmentations, lookup_dict_filename, mol_fps_filename, search_index_filename)
            for dataset in ['train', 'valid', 'test']: # rxn_smis_file_prefix = '50k_clean_rxnsmi_noreagent'
                rxn_smis_filenames[dataset] = rxn_smis_file_prefix + f'_{dataset}.pickle' 
        else:
            augmentor = None
            for dataset in ['train', 'valid', 'test']: # precomp_file_prefix = '50k_count_rdm_5'
                precomp_filenames[dataset] = precomp_file_prefix + f'_{dataset}.npz'  

        if rxn_type == 'sep':
            input_dim = rctfp_size + prodfp_size  
        elif rxn_type == 'diff':
            input_dim = rctfp_size
            assert rctfp_size == prodfp_size, 'rctfp_size must equal prodfp_size for difference fingerprints!'

        pin_memory = True if torch.cuda.is_available() else False 
        train_dataset = ReactionDataset(
            input_dim, precomp_filenames['train'], rxn_smis_filenames['train'], onthefly, augmentor)
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, 
            num_workers=self.num_workers, worker_init_fn=worker_init_fn,
            shuffle=True, pin_memory=pin_memory)
        self.train_size = len(train_dataset)
        
        val_dataset = ReactionDataset(
           input_dim, precomp_filenames['valid'], rxn_smis_filenames['valid'], onthefly, augmentor)
        self.val_loader = DataLoader(
            val_dataset, self.batch_size, 
            num_workers=self.num_workers, worker_init_fn=worker_init_fn,
            shuffle=False, pin_memory=pin_memory)
        self.val_size = len(val_dataset)
        
        test_dataset = ReactionDataset(
            input_dim, precomp_filenames['test'], rxn_smis_filenames['test'], onthefly, augmentor)
        self.test_loader = DataLoader(
            test_dataset, self.batch_size, 
            num_workers=self.num_workers, worker_init_fn=worker_init_fn,
            shuffle=False, pin_memory=pin_memory)
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset # save memory
    
    def _check_earlystop(self, current_epoch):
        self.to_break = 0
        if self.min_val_loss - self.mean_val_loss[-1] < self.min_delta:
            if self.patience <= self.wait:
                print(f'\nEarly stopped at the end of epoch: {current_epoch}, \
                train loss: {self.mean_train_loss[-1]:.4f}, top-1 train acc: {self.mean_train_acc[-1]:.4f}, \
                \nval loss: {self.mean_val_loss[-1]:.4f}, top-1 val acc: {self.mean_val_acc[-1]:. 4f}') 
                self.stats['early_stop_epoch'] = current_epoch 
                self.to_break = 1 # will break loop 
            else:
                self.wait += 1
                print('Decrease in val loss < min_delta, patience count: ', self.wait)
        else:
            self.wait = 0
            self.min_val_loss = min(self.min_val_loss, self.mean_val_loss[-1])
        
    def _update_stats(self):
        self.stats['train_time'] += (time.time() - self.start) / 60  # in minutes
        self.stats['mean_train_loss'] = self.mean_train_loss # a list with one value for each epoch 
        self.stats['mean_train_acc'] = self.mean_train_acc # a list with one value for each epoch 
        self.stats['mean_val_loss'] = self.mean_val_loss # a list with one value for each epoch
        self.stats['mean_val_acc'] = self.mean_val_acc # a list with one value for each epoch
        self.stats['min_val_loss'] = self.min_val_loss 
        self.stats['best_epoch'] = self.best_epoch
        torch.save(self.stats, self.stats_filename)

    def _checkpoint_model_and_opt(self, current_epoch: int):
        if self.distributed: 
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        checkpoint_dict = {
                            'epoch': current_epoch, # epochs are 0-indexed
                            'model_name': self.model_name,
                            'state_dict': model_state_dict,
                            'optimizer' : self.optimizer.state_dict(),
                            'stats' : self.stats
                        }
        checkpoint_filename = self.checkpoint_folder + f'{self.model_name}_{self.expt_name}_checkpoint_{current_epoch:04d}.pth.tar' 
        torch.save(checkpoint_dict, checkpoint_filename)

    def _one_batch(self, batch: tensor, backprop: bool=True):
        '''
        Passes one batch of samples through model to get scores & loss 
        Does backprop if training 

        TODO: learning rate scheduler
        '''
        # for p in self.model.parameters(): p.grad = None # faster, but less readable
        self.model.zero_grad()
        scores = self.model(batch) # size N x K 

        # positives are the 0-th index of each sample 
        loss = (scores[:, 0] + torch.logsumexp(-scores, dim=1)).mean() 

        if backprop:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        pred_labels = torch.topk(scores, 1, dim=1, largest=False)[1] # index with lowest energy is the most feasible rxn
        pred_correct = torch.where(pred_labels == 0)[0].shape[0]   # 0-th index should have the lowest energy
        return loss.item(), pred_correct
    
    def _one_epoch(self, mode='train'):
        if mode == 'train':
            self.model.train() # set model to training mode
            dataloader = self.train_loader
        elif mode == 'validate':  
            self.model.eval()
            dataloader = self.val_loader
        else: # mode == 'test'
            self.model.eval()
            dataloader = self.test_loader 

        batch_losses, batch_correct_preds = [], []
        for batch in tqdm(dataloader): 
            batch = batch.to(self.device)
            curr_batch_loss, curr_batch_correct_preds = self._one_batch(
                batch, backprop=True if mode == 'train' else False)
            batch_losses.append(curr_batch_loss)
            batch_correct_preds.append(curr_batch_correct_preds)
        return batch_losses, batch_correct_preds            

    def train(self):
        self.start = time.time()  # timeit.default_timer()     
        for epoch in range(self.begin_epoch, self.epochs):
            train_losses, train_correct_preds = self._one_epoch(mode='train')
            self.mean_train_acc.append(np.sum(train_correct_preds) / self.train_size)
            self.mean_train_loss.append(np.sum(train_losses) / self.train_size) 

            val_losses, val_correct_preds = self._one_epoch(mode='validate')
            self.mean_val_acc.append(np.sum(val_correct_preds) / self.val_size)
            self.mean_val_loss.append(np.sum(val_losses) / self.val_size)

            if self.mean_val_loss[-1] < self.min_val_loss: # track best_epoch to facilitate loading of best checkpoint 
                self.best_epoch = epoch 
 
            self._update_stats()
            if self.checkpoint:
                self._checkpoint_model_and_opt(current_epoch=epoch) 
            if self.early_stop:
                self._check_earlystop(current_epoch=epoch)
                if self.to_break: # is it time to early stop? 
                    break
                
            print(f'\nEnd of epoch: {epoch}, \
                \ntrain loss: {self.mean_train_loss[-1]:.4f}, top-1 train acc: {self.mean_train_acc[-1]:.4f}, \
                \nval loss: {self.mean_val_loss[-1]: .4f}, top-1 val acc: {self.mean_val_acc[-1]:.4f}')
 
        print(f'Total training time: {self.stats["train_time"]}')

    def test(self, saved_stats: Optional[dict]=None):
        '''
        Evaluates the model on the test set
        Parameters
        ---------
        saved_stats: Optional[dict]
            Test statistics will be stored inside this stats file 
            Used to load existing stats file when loading a trained model from checkpoint
        '''
        test_losses, test_correct_preds = self._one_epoch(mode='test')
        
        if saved_stats: 
            self.stats = saved_stats 
        assert len(self.stats.keys()) > 1, 'If loading checkpoint, you need to provide load_stats!'

        self.stats['mean_test_loss'] = np.sum(test_losses) / self.test_size
        self.stats['mean_test_acc'] = np.sum(test_correct_preds) / self.test_size
        print(f'Mean test loss: {self.stats["mean_test_loss"]}')
        print(f'Mean test top-1 accuracy: {self.stats["mean_test_acc"]}')
        torch.save(self.stats, self.stats_filename) # overrides existing training stats w/ training + test stats

    def get_scores(self, key='test', 
                   dataloader=None, dataset_len=None, 
                   print_loss=True, 
                   show_neg=False):
        ''' 
        Gets raw energy values (scores) from a trained model on a given dataloader,
        with the option to save pos_neg_smis to analyse model performance
        
        Parameters
        ----------
        key: str (Default = 'test')
            whether to get scores from train/test/val datasets
        dataloader: Optional[Dataloader] (Default = None)
            custom dataloader that loops through dataset that is not the original train, test or val 
            (which are simply accessed by providing a key parameter)
        dataset_len: Optional[int] (Default = None)
            length of dataset used for dataloader
        print_loss: bool (Default = True)
            whether to print loss over specified dataset

        Returns
        -------
        scores: Tensor
            Tensor of scores of shape (# rxns, 1 + # neg rxns)
            
        TO DO: fix show_neg: index into SMILES molecule vocab to retrieve molecules --> 
        save as groups [true product/rct SMILES, 1st NN SMILES, ... K-1'th NN SMILES])
        '''
        scores = []
        if dataloader is not None:
            assert dataset_len is not None, 'Please provide the size of the custom dataset'
            key = 'custom'
        elif key == 'test':
            dataloader = self.test_loader
        elif key == 'train':
            dataloader = self.train_loader
        elif key == 'val':
            dataloader == self.val_loader
        
        self.model.eval()
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
#                         self.trainargs['model'], self.trainargs['expt_name']))            
            else:
                for batch in tqdm(dataloader):
                    batch = batch.to(self.device)
                    scores.append(self.model(batch).cpu())
            
            scores = torch.cat(scores, dim=0).squeeze(dim=-1)
            if print_loss:
                loss = (scores[:, 0] + torch.logsumexp(-1 * scores, dim=1)).sum()
                if dataset_len is not None:
                    loss /= dataset_len
                elif key == 'test':
                    loss /= self.test_size
                elif key == 'train':
                    loss /= self.train_size
                elif key == 'val':
                    loss /= self.val_size
                print('Loss on ' + key + ':', loss.tolist())
            
#             if show_neg:
#                 return scores, pos_neg_smis
#             else:
            return scores
                # output shape N x K: 
                # where N = # positive rxns in dataset;
                # where K = 1 + # negative rxns per positive rxn

    def get_topk_acc(self, key: str='test', dataloader=None, dataset_len=None, 
                     k=1, repeats=1, 
                     save_scores=True, name_scores=None, path_scores='scores/',
                     print_results=True, print_loss=True):
        '''
        Computes top-k accuracy of trained model in classifying feasible vs infeasible chemical rxns
        (i.e. minimum energy assigned to label 0 of each training sample) 
        
        Parameters
        ----------
        save_scores: bool (Default = True)
            whether to save the generated scores Tensor
        name_scores: str (Default = None)
            filename of saved scores. Do not include '.pkl' extension. 
            Defaults to 'scores_<mode>_<key>_<expt_name>.pkl'
        
        Returns
        -------
        scores: Tensor
            Tensor of scores of shape (# rxns, 1 + # neg rxns)
            
        Also see: self.get_scores
        '''
        accs = np.array([])
        for repeat in range(repeats):
            print('Running repeat: ', repeat)
            scores = self.get_scores(key=key, dataloader=dataloader, dataset_len=dataset_len,
                                     print_loss=print_loss)
            pred_labels = torch.topk(scores, k, dim=1, largest=False)[1]
            accs = np.append(accs, torch.where(pred_labels == 0)[0].shape[0] / pred_labels.shape[0])
        
        name_scores = name_scores or 'scores_' + self.mode + '_' + key + '_' + self.expt_name
        if save_scores:
            print('Saving scores as: ', path_scores + name_scores + '.pkl')
            torch.save(scores, path_scores + name_scores + '.pkl')
            
        if print_results:
            print('Top-1 accuracies: ', accs)
            print('Avg top-1 accuracy: ', accs.mean())
            print('Variance: ', accs.var())
        return scores 