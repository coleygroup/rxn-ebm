import torch
from torch.utils.data import Dataset
import random
import pickle
from scipy import sparse 

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit import DataStructs
import numpy as np

import nmslib


def create_rxn_MorganFP_fromFP(raw_fp, num_rcts, fp_type='diff', 
                               rctfp_size=4096, prodfp_size=4096, dtype='int8'):
    '''
    fp_type: 'diff' or 'sep', 
    'diff' (difference):
    Creates reaction MorganFP following Schneider et al in J. Chem. Inf. Model. 2015, 55, 1, 39–53
    reactionFP = productFP - sum(reactantFPs)
    
    'sep' (separate):
    Creates separate reactantsFP and productFP following Gao et al in ACS Cent. Sci. 2018, 4, 11, 1465–1476
    '''
    # initialise empty fp numpy arrays
    if fp_type == 'diff':
        diff_fp = np.zeros(rctfp_size, dtype = dtype)
    elif fp_type == 'sep':
        rcts_fp = np.zeros(rctfp_size, dtype = dtype)
        prod_fp = np.zeros(prodfp_size, dtype = dtype)
    else:
        print('ERROR: fp_type not recognised!')
        return
    
    # create product FP
    try:
        fp = raw_fp[-1]
        if fp_type == 'diff':
            diff_fp += fp
        elif fp_type == 'sep':
            prod_fp = fp
    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return
                                  
    # create reactant FPs, subtracting each from product FP
    for i in range(num_rcts):
        try:
            fp = raw_fp[i]
            if fp_type == 'diff':
                diff_fp -= fp
            elif fp_type == 'sep':
                rcts_fp += fp
        except Exception as e:
            print("Cannot build reactant fp due to {}".format(e))
            return
    
    if fp_type == 'diff':
        return diff_fp
    elif fp_type == 'sep':
        return np.concatenate([rcts_fp, prod_fp])
    
def xform_rctFP(rct_FP, rct_idx, raw_fp):
    '''Helper function for cosine sampling, to swap reactant in original raw_FP with a new reactant
    '''
    outputFP = raw_fp.copy()
    outputFP[rct_idx] = rct_FP
    return outputFP

class ReactionDataset(Dataset):
    '''
    The Dataset class ReactionDataset prepares training samples of length K: 
    [pos_rxn, neg_rxn_1, ..., neg_rxn_K-1], ... where K-1 = num_neg 
    
    TO DO: remove self.num_neg for random negative sampling (use num_neg_prod & num_neg_rct)
    '''
    def __init__(self, base_path, key, trainargs, mode,
                 spaces_index=None,
                 show_neg=False, # for visualising nearest neighbors 
                 save_neg=False): # for rxn_smi (to fix later on)
        '''
        For cosine/random negative sampling:
            base_path should have the form: 'USPTO_50k_data/clean_rxn_50k_sparse_FPs', and according to key parameter,
            the correct full path will be used e.g. 'USPTO_50k_data/clean_rxn_50k_sparse_FPs_train.npz'
            
            Needs in trainargs:
                'num_neg_prod': # products to sample
                'num_neg_rct': # reactants to sample
                'cluster_path': path to search tree for cosine nearest neighbour sampling
        
        For bit corruption:
            base_path should have the form 'PATH/clean_rxn_50k_sparse_rxnFPs'
            Needs in trainargs: 
                'num_neg': # negative samples to generate
                'num_bits': # bits to randomly corrupt in rxn fingerprint 
        ''' 
        self.trainargs = trainargs
        self.mode = mode
        if self.mode == 'bit_corruption':
#             print('bit_corruption')
            self.rxn_fp = sparse.load_npz(base_path + '_' + key + '.npz')
            self.num_neg = self.trainargs['num_neg']
            assert self.num_neg > 0, 'num_neg must be positive!'
            self.num_bits = self.trainargs['num_bits']
            assert self.num_bits > 0, 'num_bits must be positive!'
            
        else:
            # print('cosine/random sampling')            
            self.fp_raw_num_rcts = sparse.load_npz(base_path + '_' + key + '.npz')            
            if 'cluster_path' in self.trainargs.keys(): # doing cosine similarity search
                if self.trainargs['cluster_path'] is not None:
                    print('_init_searchindex')
                    self._init_searchindex(spaces_index)
   
            assert 'num_neg_prod' in self.trainargs.keys() and 'num_neg_rct' in self.trainargs.keys(), \
            'You are not doing bit_corruption, but did not provide num_neg_prod and num_neg_rct!'
            self.num_neg_prod = self.trainargs['num_neg_prod']
            self.num_neg_rct = self.trainargs['num_neg_rct']
            assert self.num_neg_prod > 0 and self.num_neg_rct > 0, 'num_neg cannot be negative!' 
        
        self.fp_type = self.trainargs['fp_type']
        self.rctfp_size = self.trainargs['rctfp_size']
        self.prodfp_size = self.trainargs['prodfp_size']
        assert trainargs['rctfp_size'] == self.trainargs['prodfp_size'], \
        'rctfp_size must be equal to prodfp_size!'
#         self.save_neg = save_neg
#         self.show_neg = show_neg

    def _init_searchindex(self, spaces_index):
        self.sparseFP_vocab = sparse.load_npz(self.trainargs['sparseFP_vocab_path'])
        if self.mode == 'cosine_spaces':   
            self.clusterindex = spaces_index
            # nmslib.init(method='hnsw', space='cosinesimil_sparse', 
            #                     data_type=nmslib.DataType.SPARSE_VECTOR)
            # self.clusterindex.loadIndex(self.trainargs['cluster_path'], load_data=True)
            
            # if 'query_params' in self.trainargs.keys():
            #     self.clusterindex.setQueryTimeParams(self.trainargs['query_params'])
        else:
            with open(self.trainargs['cluster_path'], 'rb') as handle:
                self.clusterindex = pickle.load(handle)

    def random_sample_negative(self, raw_fp, num_rcts, mode):
        '''
        Randomly generates 1 negative rxn given a positive rxn fingerprint
        Returns neg_rxn_fp (fingerprint)
        ''' 
        # make this chunk of 3-4 lines into a function 
        rdm_rxn_idx = random.choice(np.arange(self.fp_raw_num_rcts.shape[0])) 
        new_fp_raw_num_rcts = self.fp_raw_num_rcts[rdm_rxn_idx].toarray()
        new_raw_fp, _ = np.split(new_fp_raw_num_rcts, [new_fp_raw_num_rcts.shape[-1]-1], axis=1)
        new_raw_fp = new_raw_fp.reshape(-1, self.rctfp_size)  
        
        if mode == 'rct':
            rct_idx = random.choice(np.arange(num_rcts))
            raw_fp[rct_idx, :] = new_raw_fp[rct_idx, :]
        else:
            raw_fp[-1, :] = new_raw_fp[-1, :] 
        return raw_fp 
    
    def hopefully_faster_vstack(self, tuple_of_arrays):
        array_1, array_2 = tuple_of_arrays
        width = array_1.shape[1]
        length_1, length_2 = array_1.shape[0], array_2.shape[0]
#         assert array_1.shape[1] == array_2.shape[1]
        array_out = np.empty((length_1 + length_2, width))
        array_out[:length_1] = array_1
        array_out[length_1:] = array_2
        
    def cosine_sample_negative(self, raw_fp, num_rcts):
        ''' 
        Replace product w/ approx nearest neighbor based on cosine similarity 
        Args:
            raw_fp: dense nparray fp of shape (max #rcts + 1, self.rctfp_size); for USPTO-50k, max #rcts = 4
            num_rcts: # rcts in current rxn corresponding to raw_fp
        
        Returns:
            list of num_neg X dense nparrays of the same shape as raw_fp, with product replaced 
            
        TODO: might be faster to sample rct_idx's for entire batch (have to design custom collate_fn)
        '''
        rct_idx = random.choice(np.arange(num_rcts)) 
        rct_prod_sparse = sparse.csr_matrix(raw_fp.copy()[[rct_idx, -1]], dtype='int8')
        
        if self.mode == 'cosine_spaces':
            nn_rct_idx_dist, nn_prod_idx_dist = self.clusterindex.knnQueryBatch(rct_prod_sparse, 
                                                                                k = max(self.num_neg_prod, self.num_neg_rct) + 1, 
                                                                                num_threads = 4) # or self.trainargs['num_threads']
            nn_rct_idxs, nn_prod_idxs = nn_rct_idx_dist[0], nn_prod_idx_dist[0]
        else:
            nn_rct_idxs, nn_prod_idxs = self.clusterindex.search(rct_prod_sparse, k=max(self.num_neg_prod, self.num_neg_rct)+1, 
                                                   return_distance=False)  #[0][1:]
            nn_rct_idxs, nn_prod_idxs = [int(idx) for idx in nn_rct_idxs], [int(idx) for idx in nn_prod_idxs]
        nn_prod_FPs = [self.sparseFP_vocab[idx].toarray() for idx in nn_prod_idxs[1: self.num_neg_prod + 1]]
        nn_rct_FPs = [self.sparseFP_vocab[idx].toarray() for idx in nn_rct_idxs[1: self.num_neg_rct + 1]] # 1 x 4096 nparray
        
        # try pre-allocating output list in memory, to avoid list concatenation (which might be slow)
        out_FPs = [0] * (self.num_neg_prod + self.num_neg_rct)
        for i, prod_FP in enumerate(nn_prod_FPs):
            out_FPs[i] = np.vstack((raw_fp[:-1], prod_FP))
        for j, rct_FP in enumerate(nn_rct_FPs):
            out_FPs[self.num_neg_prod + j] = xform_rctFP(rct_FP, rct_idx, raw_fp)
        return out_FPs
#         if self.show_neg:
#             return [np.vstack((raw_fp[:-1], FP)) for FP in nn_prod_FPs] + [xform_rctFP(rctFP, rct_idx, raw_fp) for rctFP in nn_rct_FPs], nn_prod_indices, nn_rct_indices
#         else:
#             return [np.vstack((raw_fp[:-1], FP)) for FP in nn_prod_FPs] + [xform_rctFP(rctFP, rct_idx, raw_fp) for rctFP in nn_rct_FPs]  
            
    def random_bit_corrupter(self, rxn_fp, num_bits=10):
        ''' Randomly selects <num_bits> bits in rxn_fp & randomly replaces them w/ -1, 0 or 1
        Args: 
            rxn_fp: reaction fingerprint that can be indexed (usu. np array)
            NOTE: Please use rxn_fp.copy() to avoid modifying your original rxn_fp
        Returns:
            corrupted fp of same shape and type as rxn_fp
        '''
        rdm_idx = random.sample(range(rxn_fp.shape[-1]), k=num_bits)
        rxn_fp[0, rdm_idx] = [random.choice([-1, 0, 1]) for bit in rxn_fp[0, rdm_idx]]
        return rxn_fp
    
    def __getitem__(self, idx):
        ''' 
        Returns 1 training sample in the form [pos_rxn_fp, neg_rxn_1_fp, ..., neg_rxn_K-1_fp]
        num_neg / num_neg_rct & num_neg_prod: a hyperparameter to be tuned
        '''
        if torch.is_tensor(idx): 
            idx = idx.tolist() 
        
        if self.mode == 'bit_corruption':
            pos_rxn_fp = self.rxn_fp[idx].toarray()
            neg_rxn_fps = [self.random_bit_corrupter(pos_rxn_fp.copy(), self.num_bits) 
                           for i in range(self.num_neg)]
            return torch.Tensor([pos_rxn_fp, *neg_rxn_fps])
        
        else:
            fp_raw_num_rcts = self.fp_raw_num_rcts[idx].toarray() 
            pos_raw_fp, num_rcts = np.split(fp_raw_num_rcts, [fp_raw_num_rcts.shape[-1]-1], axis=1) 
            pos_raw_fp = pos_raw_fp.reshape(-1, self.rctfp_size) 
            num_rcts = num_rcts[0][0]
            pos_rxn_fp = create_rxn_MorganFP_fromFP(pos_raw_fp.copy(), num_rcts, fp_type=self.fp_type, 
                                                    rctfp_size=self.rctfp_size, prodfp_size=self.prodfp_size)
            
            if 'cluster_path' in self.trainargs.keys():
                if self.trainargs['cluster_path'] is not None: # cosine similarity search
#                 if self.show_neg:
#                     neg_raw_fps, nn_prod_indices, nn_rct_indices = self.cosine_sample_negative(pos_raw_fp, num_rcts)
#                 else:
                    neg_raw_fps = self.cosine_sample_negative(pos_raw_fp, num_rcts)
            else:
                neg_raw_fps = [0] * (self.num_neg_prod + self.num_neg_rct)
                for i in range(self.num_neg_prod):
                    neg_raw_fps[i] = self.random_sample_negative(pos_raw_fp.copy(), num_rcts, 'prod')
                for j in range(self.num_neg_rct):
                    neg_raw_fps[self.num_neg_prod + j] = self.random_sample_negative(pos_raw_fp.copy(), num_rcts, 'rct') 
#                 neg_raw_fps = [self.random_sample_negative(pos_raw_fp.copy(), num_rcts, 'prod') 
#                                for i in range(self.num_neg_prod)] + [
#                                self.random_sample_negative(pos_raw_fp.copy(), num_rcts, 'rct') 
#                                for i in range(self.num_neg_rct)]

            neg_rxn_fps = [create_rxn_MorganFP_fromFP(neg_raw_fp.copy(), num_rcts, fp_type=self.fp_type, 
                                                      rctfp_size=self.rctfp_size, prodfp_size=self.prodfp_size)
                            for neg_raw_fp in neg_raw_fps]
#             if self.show_neg:
#                 return torch.Tensor([pos_rxn_fp, *neg_rxn_fps]), nn_prod_indices, nn_rct_indices
#             else:
            return torch.Tensor([pos_rxn_fp, *neg_rxn_fps])

    def __len__(self):
        if self.mode == 'bit_corruption':
            return self.rxn_fp.shape[0]
        else:
            return self.fp_raw_num_rcts.shape[0]