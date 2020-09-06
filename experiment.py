import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import time
import random

from model.utils import (save_checkpoint, seed_everything)
from data.data import ReactionDataset

class Experiment():
    '''
    epochs are 1-indexed (i.e. start from 1, 2, 3 ... not 0, 1, 2 ...)
    if load_checkpoint == True, load_optimizer, load_stats & begin_epoch must be provided 
    '''
    def __init__(self, model, trainargs, mode=None,
                 load_optimizer=None, load_checkpoint=False, load_stats=None, begin_epoch=None):
        self.device = trainargs['device']
        model = model.to(self.device)
        self.model = model
        self.trainargs = trainargs 
        self.best_epoch = None # will be automatically assigned after 1 epoch
        self.mode = mode # for bit corruption vs cosine/random sampling
        
        if load_checkpoint: 
            assert load_optimizer is not None, 'load_checkpoint requires load_optimizer!'
            self.optimizer = load_optimizer # load optimizer w/ state dict from checkpoint
            
            assert load_stats is not None, 'load_checkpoint requires load_stats!'
            self.stats = load_stats
            self.mean_train_loss = self.stats['mean_train_loss']
            self.min_val_loss = self.stats['min_val_loss']
            self.mean_val_loss = self.stats['mean_val_loss']
            try:
              self.mean_train_acc = self.stats['mean_train_acc']
              self.mean_val_acc = self.stats['mean_val_acc']
            except Exception as e:
              print(e)
            
            assert begin_epoch is not None, 'load_checkpoint requires begin_epoch!'
            self.begin_epoch = begin_epoch

        else: # init fresh optimizer 
            self.optimizer = trainargs['optimizer'](model.parameters(), lr=trainargs['learning_rate'])
            
            self.mean_train_loss = []
            self.mean_train_acc = []
            self.min_val_loss = 1e9
            self.mean_val_loss = []
            self.mean_val_acc = []
            self.begin_epoch = 1
            self.stats = {'trainargs': self.trainargs, 'train_time': 0} # to store training statistics  

        self.pin_memory = True if torch.cuda.is_available() else False
        train_dataset = ReactionDataset(trainargs['base_path'], 'train', trainargs=self.trainargs, mode=mode)
        self.train_loader = DataLoader(train_dataset, trainargs['batch_size'], shuffle=True, pin_memory=self.pin_memory)
        self.train_size = len(train_dataset)
        
        val_dataset = ReactionDataset(trainargs['base_path'], 'valid', trainargs=self.trainargs, mode=mode)
        self.val_loader = DataLoader(val_dataset, 2 * trainargs['batch_size'], shuffle=False, pin_memory=self.pin_memory)
        self.val_size = len(val_dataset)
        
        test_dataset = ReactionDataset(self.trainargs['base_path'], 'test', trainargs=self.trainargs, mode=mode)
        self.test_loader = DataLoader(test_dataset, 2 * self.trainargs['batch_size'], shuffle=False, pin_memory=self.pin_memory)
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset # save memory

        seed_everything(trainargs['random_seed'])
    
    def train_one(self, batch, val=False):
        '''
        Trains model for 1 epoch
        TO DO: learning rate scheduler + logger 
        '''
        # self.model.zero_grad()
        for p in self.model.parameters(): p.grad = None
        scores = self.model(batch) # size N x K 

        softmax = nn.Softmax(dim=1) 
        probs = torch.clamp(softmax(scores), min=1e-12) # size N x K, clamped to >= 1e-12 for safe log 

        # positives are the 0-th index of each sample 
        loss = -torch.log(probs[:, 0]).sum() # probs[:, 0] is size N x 1 --> sum/mean to 1 value

        if not val:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        pred_labels = torch.topk(scores, 1, dim=1)[1]
        pred_correct = torch.where(pred_labels == 0)[0].shape[0]  
        return loss.item(), pred_correct

    def train(self):
        '''
        Trains model for epochs provided in trainargs
        Currently supports feed-forward networks: 
            FF_diff: takes as input a difference FP of fp_size & fp_radius
            FF_sep: takes as input a concatenation of [reactants FP, product FP] 
        '''
        start = time.time()
        to_break = 0 # whether to early stop & break loop after saving checkpoint
        
        for epoch in np.arange(self.begin_epoch, self.trainargs['epochs']): # epochs are 1-indexed (as of 27th Aug 2 am)
            self.model.train() # set model to training mode
            train_loss, train_pred_correct = [], []
            for batch in tqdm(self.train_loader): 
                batch = batch.to(self.device)
                batch_train_loss, batch_pred_correct = self.train_one(batch, val=False)
                train_loss.append(batch_train_loss)
                train_pred_correct.append(batch_pred_correct)
                del batch

            self.mean_train_acc.append(np.sum(train_pred_correct) / self.train_size)
            self.mean_train_loss.append(np.sum(train_loss) / self.train_size) 

            self.model.eval() # validation mode
            val_loss, val_pred_correct = [], []
            with torch.no_grad():
                for batch in tqdm(self.val_loader):
                    batch = batch.to(self.device)
                    batch_val_loss, batch_pred_correct = self.train_one(batch, val=True)
                    val_loss.append(batch_val_loss)
                    val_pred_correct.append(batch_pred_correct)
                    del batch
                
                self.mean_val_acc.append(np.sum(val_pred_correct) / self.val_size)
                self.mean_val_loss.append(np.sum(val_loss) / self.val_size)
                
                if self.trainargs['early_stop'] and \
                self.min_val_loss - self.mean_val_loss[-1] < self.trainargs['min_delta']:
                    if self.trainargs['patience'] <= wait:
                        print('\nEarly stopped at the end of Epoch: {}, train_loss: {}, train_acc: {}, \nval_loss: {}, val_acc: {}'.format(
                                             epoch, 
                                             np.around(self.mean_train_loss[-1], decimals=4), 
                                             np.around(self.mean_train_acc[-1], decimals=4),
                                             np.around(self.mean_val_loss[-1], decimals=4),
                                             np.around(self.mean_val_acc[-1], decimals=4)
                                             ) )
                        self.stats['early_stop_epoch'] = epoch 
                        to_break = 1 # will break loop after saving checkpoint
                    else:
                        wait += 1
                        print('Decrease in val loss < min_delta, patience count: ', wait)
                else:
                    wait = 0
                    self.min_val_loss = min(self.min_val_loss, self.mean_val_loss[-1])
                
                if self.mean_val_loss[-1] < self.min_val_loss:
                    self.best_epoch = epoch # track best_epoch to load best_checkpoint 

            self.stats['mean_train_loss'] = self.mean_train_loss
            self.stats['mean_train_acc'] = self.mean_train_acc
            self.stats['mean_val_loss'] = self.mean_val_loss
            self.stats['mean_val_acc'] = self.mean_val_acc
            self.stats['min_val_loss'] = self.min_val_loss
            self.stats['best_epoch'] = self.best_epoch

            if self.trainargs['checkpoint']: # adapted from moco: main_moco.py
                save_checkpoint({
                        'epoch': epoch, # epochs are 1-indexed
                        'model': self.trainargs['model'],
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'stats' : self.stats,
                    }, is_best=False, 
                    filename=self.trainargs['checkpoint_path']+'{}_{}_checkpoint_{:04d}.pth.tar'.format(
                        self.trainargs['model'], self.trainargs['expt_name'], epoch))
                # checkpoint stats also 
                torch.save(self.stats, self.trainargs['checkpoint_path']+'{}_{}_stats.pkl'.format(
                      self.trainargs['model'], self.trainargs['expt_name']))
            
            if to_break:
                break
                
            print('\nEpoch: {}, train_loss: {}, train_acc: {}, \nval_loss: {}, val_acc: {}'.format(epoch, 
                                             np.around(self.mean_train_loss[-1], decimals=4), 
                                             np.around(self.mean_train_acc[-1], decimals=4),
                                             np.around(self.mean_val_loss[-1], decimals=4),
                                             np.around(self.mean_val_acc[-1], decimals=4)
                                             ) )
            
        self.stats['train_time'] += (time.time() - start) / 60
        torch.save(self.stats, self.trainargs['checkpoint_path']+'{}_{}_stats.pkl'.format(
            self.trainargs['model'], self.trainargs['expt_name']))   # save final training stats

    def test(self, load_stats=None):
        '''
        Evaluates the model on the test set
        '''
        test_loss_to_sum, test_loss, test_pred_correct = [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = batch.to(self.device)
                batch_test_loss, batch_pred_correct = self.train_one(batch, val=True)
                test_loss_to_sum.append(batch_test_loss)
                test_loss.append(test_loss_to_sum[-1] / len(batch))
                test_pred_correct.append(batch_pred_correct)
                del batch
        
        if load_stats is not None: 
            self.stats = load_stats 
        assert len(self.stats.keys()) > 1, 'If loading checkpoint, you need to provide load_stats!'
        
        self.stats['test_loss'] = test_loss 
        self.stats['mean_test_loss'] = np.sum(test_loss_to_sum) / self.test_size
        self.stats['mean_test_acc'] = np.sum(test_pred_correct) / self.test_size
        print('train_time: {}'.format(self.stats['train_time']))
        print('test_loss: {}'.format(self.stats['test_loss']))
        print('mean_test_loss: {}'.format(self.stats['mean_test_loss']))
        print('mean_test_acc: {}'.format(self.stats['mean_test_acc']))

        # overrides existing training stats w/ training + test stats
        torch.save(self.stats, self.trainargs['checkpoint_path']+'{}_{}_stats.pkl'.format(
            self.trainargs['model'], self.trainargs['expt_name']))

    def get_scores(self, dataloader, save_neg=False, get_loss=True):
        ''' 
        Gets raw energy values (scores) from a trained model on a given dataloader,
        with the option to save pos_neg_smis to analyse model performance
        
        TO DO: fix save_neg: index into SMILES molecule vocab to retrieve molecules --> 
        save as groups [true product/rct SMILES, 1st NN SMILES, ... K-1'th NN SMILES])
        '''
        scores = []
        self.model.eval()
        with torch.no_grad():
            if save_neg:      # save neg rxn_smis to analyse model performance           
                pos_neg_smis = []
                for pos_neg_smi, batch in tqdm(dataloader):
                    batch = batch.to(self.device)
                    scores.append(self.model(batch).cpu()) # scores: size N x K 
                    pos_neg_smis.append(pos_neg_smi)
                    del batch
                    
                torch.save(pos_neg_smis, self.trainargs['checkpoint_path']+'{}_{}_posnegsmi.pkl'.format(
                        self.trainargs['model'], self.trainargs['expt_name']))
                
                return torch.cat(scores, dim=0).squeeze(dim=-1), pos_neg_smis
            
            else:
                for batch in tqdm(dataloader):
                    batch = batch.to(self.device)
                    scores.append(self.model(batch).cpu())
                    del batch
                    
                if get_loss:
                    softmax = nn.Softmax(dim=1) 
                    probs = torch.clamp(softmax(scores), min=1e-12) # size N x K, clamped to >= 1e-12 for safe log 

                    # positives are the 0-th index of each sample 
                    loss = -torch.log(probs[:, 0]).sum() # probs[:, 0] is size N x 1 --> sum/mean to 1 value
                    print(loss.item())
                    
                return torch.cat(scores, dim=0).squeeze(dim=-1)
            # output shape N x K, 
            # N = # positive rxns in dataset
            # K = 1 + # negative rxns per positive rxn

    def get_topk_acc(self, dataloader, k=1, repeats=1, toprint=True, get_loss=True):
        '''
        Computes top-k accuracy of trained model in classifying feasible vs infeasible chemical rxns
        (i.e. maximum score assigned to label 0 of each training sample) 
        
        Returns: Tensor of scores 
        '''
        accs = np.array([])
        for repeat in range(repeats):
            scores = self.get_scores(dataloader, get_loss=get_loss)
            pred_labels = torch.topk(scores, k, dim=1)[1]
            accs = np.append(accs, torch.where(pred_labels == 0)[0].shape[0] / pred_labels.shape[0])
            
        if toprint:
            print('Top-1 accuracies: ', accs)
            print('Avg top-1 accuracy: ', accs.mean())
            print('Variance: ', accs.var())

        return scores # (accs, accs.mean(), accs.var())