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
                 load_optimizer=None, load_checkpoint=False, 
                 load_stats=None, begin_epoch=None):
        self.device = trainargs['device']
        model = model.to(self.device)
        self.model = model
        self.trainargs = trainargs 
        self.best_epoch = None # will be automatically assigned after 1 epoch
        self.mode = mode # for bit corruption vs cosine/random sampling
        print('Initialising experiment ', self.trainargs['expt_name'], '\nwith mode: ', mode)
        
        if load_checkpoint: 
            self._load_checkpoint(load_optimizer, load_stats, begin_epoch)

        else: # init fresh optimizer 
            self._init_opt_and_stats()

        self._init_dataloaders(mode) 

        seed_everything(self.trainargs['random_seed'])

    def __repr__(self):
        return 'Experiment object running on mode: ' + self.mode
    
    def _load_checkpoint(self, load_optimizer, load_stats, begin_epoch):
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
    
    def _init_opt_and_stats(self):
        self.optimizer = self.trainargs['optimizer'](self.model.parameters(), 
                                                    lr=self.trainargs['learning_rate'])
        # to store training statistics  
        self.mean_train_loss = []
        self.mean_train_acc = []
        self.min_val_loss = 1e9
        self.mean_val_loss = []
        self.mean_val_acc = []
        self.begin_epoch = 1
        self.wait = 0 # counter for _check_earlystop()
        self.stats = {'trainargs': self.trainargs, 'train_time': 0} 
        self.stats_filename = self.trainargs['checkpoint_path'] + \
                            '{}_{}_stats.pkl'.format(self.trainargs['model'],
                                                    self.trainargs['expt_name'])
    
    def _init_dataloaders(self, mode):
        self.pin_memory = True if torch.cuda.is_available() else False
        train_dataset = ReactionDataset(self.trainargs['base_path'], 'train', 
                                        trainargs=self.trainargs, mode=mode)
        self.train_loader = DataLoader(train_dataset, self.trainargs['batch_size'], 
                                        shuffle=True, pin_memory=self.pin_memory)
        self.train_size = len(train_dataset)
        
        val_dataset = ReactionDataset(self.trainargs['base_path'], 'valid', 
                                        trainargs=self.trainargs, mode=mode)
        self.val_loader = DataLoader(val_dataset, 2 * self.trainargs['batch_size'], 
                                        shuffle=False, pin_memory=self.pin_memory)
        self.val_size = len(val_dataset)
        
        test_dataset = ReactionDataset(self.trainargs['base_path'], 'test', 
                                        trainargs=self.trainargs, mode=mode)
        self.test_loader = DataLoader(test_dataset, 2 * self.trainargs['batch_size'], 
                                        shuffle=False, pin_memory=self.pin_memory)
        self.test_size = len(test_dataset)
        del train_dataset, val_dataset, test_dataset # save memory


    def train_one(self, batch, val=False):
        '''
        Trains model for 1 epoch
        TO DO: learning rate scheduler + logger 
        '''
        # self.model.zero_grad()
        for p in self.model.parameters(): p.grad = None
        scores = self.model(batch) # size N x K 

        # positives are the 0-th index of each sample 
        loss = (scores[:, 0] + torch.logsumexp(-scores, dim=1)).sum() 

        if not val:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        pred_labels = torch.topk(scores, 1, dim=1)[1]
        pred_correct = torch.where(pred_labels == 0)[0].shape[0]  
        return loss.item(), pred_correct
    
    def _check_earlystop(self, current_epoch):
        self.to_break = 0
        if self.min_val_loss - self.mean_val_loss[-1] < self.trainargs['min_delta']:
            if self.trainargs['patience'] <= self.wait:
                print('\nEarly stopped at the end of epoch: {}, \
                train loss: {}, top-1 train acc: {}, \
                \nval loss: {}, top-1 val acc: {}'.format(current_epoch, 
                                            np.around(self.mean_train_loss[-1], decimals=4), 
                                            np.around(self.mean_train_acc[-1], decimals=4),
                                            np.around(self.mean_val_loss[-1], decimals=4),
                                            np.around(self.mean_val_acc[-1], decimals=4)
                                            ) )
                self.stats['early_stop_epoch'] = current_epoch 
                self.to_break = 1 # will break loop 
            else:
                self.wait += 1
                print('Decrease in val loss < min_delta, patience count: ', self.wait)
        else:
            self.wait = 0
            self.min_val_loss = min(self.min_val_loss, self.mean_val_loss[-1])
        
    def _save_stats(self):
        self.stats['mean_train_loss'] = self.mean_train_loss
        self.stats['mean_train_acc'] = self.mean_train_acc
        self.stats['mean_val_loss'] = self.mean_val_loss
        self.stats['mean_val_acc'] = self.mean_val_acc
        self.stats['min_val_loss'] = self.min_val_loss
        self.stats['best_epoch'] = self.best_epoch

        torch.save(self.stats, self.stats_filename)

    def _checkpoint_model_and_opt(self, current_epoch):
        checkpoint_dict = {
                                    'epoch': current_epoch, # epochs are 1-indexed
                                    'model': self.trainargs['model'],
                                    'state_dict': self.model.state_dict(),
                                    'optimizer' : self.optimizer.state_dict(),
                                    'stats' : self.stats,
                                }
        checkpoint_filename = self.trainargs['checkpoint_path'] + \
                    '{}_{}_checkpoint_{:04d}.pth.tar'.format(
                                                            self.trainargs['model'], 
                                                            self.trainargs['expt_name'], 
                                                            current_epoch)
        torch.save(checkpoint_dict, checkpoint_filename)

    def train(self):
        '''
        Trains model for epochs provided in trainargs
        '''
        start = time.time()        
        # epochs are 1-indexed
        for epoch in np.arange(self.begin_epoch, self.trainargs['epochs']):
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
            # track best_epoch to facilitate loading of best checkpoint 
            if self.mean_val_loss[-1] < self.min_val_loss:
                self.best_epoch = epoch 
                
            self._save_stats()
            if self.trainargs['checkpoint']:
                self._checkpoint_model_and_opt(current_epoch=epoch) 
            if self.trainargs['early_stop']:
                self._check_earlystop(current_epoch=epoch)
                if self.to_break: # is it time to early stop? 
                    break
                
            print('\nEnd of epoch: {}, \
                   \ntrain loss: {}, top-1 train acc: {}, \
                   \nval loss: {}, top-1 val acc: {}'.format(epoch, 
                                             np.around(self.mean_train_loss[-1], decimals=4), 
                                             np.around(self.mean_train_acc[-1], decimals=4),
                                             np.around(self.mean_val_loss[-1], decimals=4),
                                             np.around(self.mean_val_acc[-1], decimals=4)
                                             ) )
            
         # save final training stats
        self.stats['train_time'] += (time.time() - start) / 60
        print('Total training time: {}'.format(self.stats['train_time']))
        torch.save(self.stats, self.stats_filename)  

    def test(self, load_stats=None):
        '''
        Evaluates the model on the test set
        Parameters
        ---------
        load_stats: reference to a .pkl stats dictionary (default None)
            Test statistics will be stored inside this stats file 
            Used to load existing stats file when loading a trained model from checkpoint
        '''
        test_losses_to_sum, mean_batch_test_losses, test_pred_correct = [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = batch.to(self.device)
                batch_test_loss, batch_pred_correct = self.train_one(batch, val=True)
                test_losses_to_sum.append(batch_test_loss)
                mean_batch_test_losses.append(batch_test_loss / len(batch))
                test_pred_correct.append(batch_pred_correct)
                del batch
        
        if load_stats: 
            self.stats = load_stats 
        assert len(self.stats.keys()) > 1, 'If loading checkpoint, you need to provide load_stats!'
        
        self.stats['test_loss'] = mean_batch_test_losses 
        self.stats['mean_test_loss'] = np.sum(test_losses_to_sum) / self.test_size
        self.stats['mean_test_acc'] = np.sum(test_pred_correct) / self.test_size

        print('Mean test loss: {}'.format(self.stats['mean_test_loss']))
        print('Mean test top-1 accuracy: {}'.format(self.stats['mean_test_acc']))

        # overrides existing training stats w/ training + test stats
        torch.save(self.stats, self.stats_filename)


    def get_scores(self, key='test', 
                   dataloader=None, dataset_len=None, 
                   print_loss=True, 
                   save_neg=False):
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
            
        TO DO: fix save_neg: index into SMILES molecule vocab to retrieve molecules --> 
        save as groups [true product/rct SMILES, 1st NN SMILES, ... K-1'th NN SMILES])
        '''
        scores = []
        if dataloader is not None:
            assert dataset_len is not None, 'Please provide size of custom dataset'
            key = 'custom'
        elif key == 'test':
            dataloader = self.test_loader
        elif key == 'train':
            dataloader = self.train_loader
        elif key == 'val':
            dataloader == self.val_loader
        
        self.model.eval()
        with torch.no_grad():
            if save_neg:  # save neg rxn_smis to analyse model performance   
                print('save_neg is not functional now!')
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
                    del batch
            
            scores = torch.cat(scores, dim=0).squeeze(dim=-1)
            if print_loss:
                loss = scores[:, 0] + torch.logsumexp(-1 * scores, dim=1).sum()
                if dataset_len is not None:
                    loss /= dataset_len
                elif key == 'test':
                    loss /= self.test_size
                elif key == 'train':
                    loss /= self.train_size
                elif key == 'val':
                    loss /= self.val_size
                print('Loss on ', key, loss)
            
            # if save_neg:
            #     return scores, pos_neg_smis
            # else:
            return scores
                # output shape N x K: 
                # where N = # positive rxns in dataset;
                # where K = 1 + # negative rxns per positive rxn

    def get_topk_acc(self, key='test', dataloader=None, dataset_len=None, 
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
        
        name_scores = name_scores or 'scores_' + self.mode + '_' + key + '_' + self.trainargs['expt_name']
        if save_scores:
            print('Saving scores as: ', path_scores + name_scores, '.pkl')
            torch.save(scores, path_scores + name_scores + '.pkl')
            
        if print_results:
            print('Top-1 accuracies: ', accs)
            print('Avg top-1 accuracy: ', accs.mean())
            print('Variance: ', accs.var())

        return scores # (accs, accs.mean(), accs.var())