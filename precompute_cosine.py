from torch.utils.data import Dataset
import random
from scipy import sparse 
import numpy as np

import nmslib

class CosineSimilarity(Dataset): 
    '''
    Uses nmslib's sparse cosine similarity search index to fetch nearest neighbour 
    molecules of reactants & products in our ReactionDataset & returns them as indices

    These indices point to the corresponding sparse fingerprint in the sparse vocab file
    This pre-computation avoids having to repeat the expensive kNN querying operation every epoch
    
    Returns
    -------
    __getitem__(index) : List[int]
        a list of the following in order: 
        1) nearest neighbour reactant indices * aug_args['num_neg_rct'] 
        2) nearest neighbour product indices * aug_args['num_neg_prod']
        3) an index (between 0 to max # of reactants in a rxn in the dataset) 
           that tells us which reactant molecule in this rxn is to be replaced 
    
    TODO: precompute the whole reaction fingerprint for negative examples and store them in sparse format? 
    OR: precompute once every N epochs? where N is a hyperparameter to be tuned 
    pros: 
        this would speed up dataloading even further 
    cons: 
        if num_neg_rct & num_neg_prod are large, this may result in huge files on disk (and memory), 
        necessitating streaming via hdf5 for example
        loses the epoch-on-epoch stochasticity for bit corruption & random sampling
    '''
    def __init__(self, base_path, key, aug_args): 
        self.aug_args = aug_args
        self.fp_raw_num_rcts = sparse.load_npz(base_path + '_' + key + '.npz')
        self.fp_raw_num_rcts_length = self.fp_raw_num_rcts.shape[0]    
        self.rctfp_size = self.aug_args['rctfp_size']       
        self.max_num_rcts = self.fp_raw_num_rcts[0].toarray()[:, :-1].shape[1] // self.rctfp_size 
        self.num_neg_prod = self.aug_args['num_neg_prod']
        self.num_neg_rct = self.aug_args['num_neg_rct']

        self.clusterindex = None  # if multiprocessing, this will be initialised from worker_init_fn
        if self.aug_args['num_workers'] == 0:
            self._load_nmslib()

    def _load_nmslib(self): 
        self.clusterindex = nmslib.init(method='hnsw', space='cosinesimil_sparse', 
                            data_type=nmslib.DataType.SPARSE_VECTOR)
        self.clusterindex.loadIndex(self.aug_args['cluster_path'], load_data=True)
        if 'query_params' in self.aug_args.keys():
            self.clusterindex.setQueryTimeParams(self.aug_args['query_params'])
    
    def cosine_sample_negative(self, raw_fp, num_rcts):
        if self.clusterindex is None: # safety check
            self._load_nmslib()

        rct_idx = random.randint(0, num_rcts - 1) 
        rct_prod_sparse = sparse.csr_matrix(raw_fp.copy()[[rct_idx, -1]], dtype='int8')
        nn_rct_idx_dist, nn_prod_idx_dist = self.clusterindex.knnQueryBatch(
                                                    rct_prod_sparse, k = max(self.num_neg_prod, self.num_neg_rct) + 1, 
                                                     num_threads = self.aug_args['num_threads'])   
        nn_rct_idxs, nn_prod_idxs = nn_rct_idx_dist[0][1: self.num_neg_rct + 1], nn_prod_idx_dist[0][1: self.num_neg_prod + 1]
        
        out = [0] * (self.num_neg_rct + self.num_neg_prod + 1)
        out[ : self.num_neg_rct] = nn_rct_idxs
        out[self.num_neg_rct : -1] = nn_prod_idxs
        out[-1] = rct_idx
        return out
 
    def __getitem__(self, idx):
        fp_raw_num_rcts = self.fp_raw_num_rcts[idx].toarray() 
        pos_raw_fp = fp_raw_num_rcts[:, :-1].reshape(self.max_num_rcts, self.rctfp_size) 
        num_rcts = fp_raw_num_rcts[0, -1]
        return self.cosine_sample_negative(pos_raw_fp, num_rcts)

    def __len__(self):
        return self.fp_raw_num_rcts_length

def main():
    # from torch.utils.data import DataLoader
    import torch
    import pickle 
    from tqdm import tqdm
    from experiment.utils import setup_paths

    LOCAL = True # CHANGE THIS 
    mode = 'cosine_spaces' # CHANGE THIS 
    expt_name = 'cosine_spaces_5+5' # CHANGE THIS 

    checkpoint_folder, base_path, cluster_path, sparseFP_vocab_path = setup_paths(expt_name, mode, LOCAL)

    aug_args = {
    'num_workers': 0, # 0 or 4
    'random_seed': 0, # affects neg rxn sampling since it is random
    
    'batch_size': 1024,
    'rctfp_size': 4096,     
    'num_neg_prod': 5, 
    'num_neg_rct': 5,
    # 'num_bits': 4, 
    # 'num_neg': 5,
    
    'base_path': base_path, # refer to top of notebook 
    'cluster_path': cluster_path,
    'query_params': {'efSearch': 100}, # for nmslib, good value to hit 96% recall
    'num_threads': 4, # for nmslib
 
    'expt_name': expt_name,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }   
    
    keys = ['train', 'valid', 'test'] 
    for key in keys:
        print('processing: ', key)
        nnindices_rctidx = []
        dataset = CosineSimilarity(aug_args['base_path'], key, aug_args=aug_args)
        for i in tqdm(range(len(dataset))):
            # print(train_dataset.__getitem__(i))
            # if i > 10: break
            nnindices_rctidx.append(dataset.__getitem__(i))

        # train_loader = DataLoader(train_dataset, batch_size=aug_args['batch_size'], num_workers=aug_args['num_workers'],
        #                           worker_init_fn=worker_init_fn, shuffle=False)

        with open('USPTO_50k_data/{}_{}.pkl'.format(expt_name, key), 'wb') as handle:
            pickle.dump(nnindices_rctidx, handle)

if __name__ == '__main__': 
    main() 