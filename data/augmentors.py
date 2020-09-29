import torch
import torch.tensor as tensor
from torch.utils.data import Dataset
import random
import pickle
import scipy
from scipy import sparse 
import numpy as np
import nmslib
from typing import List, Optional, Tuple, Union
from pathlib import Path

sparse_fp = scipy.sparse.csr_matrix 
dense_fp = np.ndarray # try not to use, more memory intensive

def rcts_prod_fps_from_rxn_smi(rxn_smi: str, fp_type: str, lookup_dict: dict, 
                              mol_fps: sparse_fp) -> Tuple[sparse_fp, sparse_fp]:
    ''' No need to check for KeyError in lookup_dict
    '''
    prod_smi = rxn_smi.split('>')[-1]
    prod_idx = lookup_dict[prod_smi]
    prod_fp = mol_fps[prod_idx]

    if fp_type == 'bit':
        dtype = 'bool' # to add rcts_fp w/o exceeding value of 1
        final_dtype = 'int8' # to cast rcts_fp back into 'int8'
    else: # count
        dtype = 'int16' # infer dtype
        final_dtype = dtype

    rcts_smis = rxn_smi.split('>')[0].split('.')
    for i, rct_smi in enumerate(rcts_smis):
        rct_idx = lookup_dict[rct_smi]
        if i == 0:
            rcts_fp = mol_fps[rct_idx].astype(dtype)
        else:
            rcts_fp = rcts_fp + mol_fps[rct_idx].astype(dtype)
    return rcts_fp.astype(final_dtype), prod_fp

def make_rxn_fp(rcts_fp: sparse_fp, prod_fp: sparse_fp, rxn_type: str = 'diff') -> sparse_fp:
    ''' 
    Assembles rcts_fp (a sparse array of 1 x fp_size) & prod_fp (another sparse array, usually of the same shape)
    into a reaction fingerprint, rxn_fp, of the fingerprint type requested
    
    rxn_type : str (Default = 'diff')
        currently supports 'diff' & 'sep' fingerprints
    '''
    if rxn_type == 'diff':
        rxn_fp = prod_fp - rcts_fp
    elif rxn_type == 'sep':
        rxn_fp = sparse.hstack([rcts_fp, prod_fp])
    return rxn_fp 

class CosineAugmentor:
    '''
    Generates negative reaction fingerprints (on-the-fly) by fetching nearest neighbours 
    (by cosine similarity) of product molecules in a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants
    
    TODO: Currently made for fingerprints - think about how to make this a base class 
    for both fingerprints and graphs. 
    TODO: Allow saving of indices (or the negative fingerprints themselves) on disk
    '''
    def __init__(self, num_neg: int, lookup_dict: dict, 
                mol_fps: sparse_fp, search_index: nmslib.dist.FloatIndex, 
                rxn_type: str='diff', fp_type: str='count', num_threads: int=4):
        self.num_neg = num_neg
        self.lookup_dict = lookup_dict
        self.mol_fps = mol_fps
        self.search_index = search_index # initialised from elsewhere? prolly worker_init_fn
        self.num_threads = num_threads
        self.rxn_type = rxn_type
        self.fp_type = fp_type
    
    def gen_one_sample(self, rxn_smi):
        ''' 
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)

        nn_prod_idxs_with_dist = self.search_index.knnQueryBatch(prod_fp, k=self.num_neg+1, num_threads=self.num_threads)
        # print('nn_prod_idxs with dist: ', nn_prod_idxs_with_dist)
        nn_prod_idxs = nn_prod_idxs_with_dist[0][0] # only 1  element, so need additional [0] to access it
        # then, first element of the tuple is the np.ndarray of indices
        # print('nn_prod_idxs: ', nn_prod_idxs)
        # don't take the first index (0) because that's the original molecule! 
        nn_prod_fps = [ self.mol_fps[idx] for idx in nn_prod_idxs[1: self.num_neg + 1] ]
        neg_rxn_fps = []
        for nn_prod_fp in nn_prod_fps:
            neg_rxn_fp = make_rxn_fp(rcts_fp, nn_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp) 
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)
        return neg_rxn_fps # torch.Tensor([pos_rxn_fp, *neg_rxn_fps])

class RandomAugmentor:
    '''
    Generates negative reaction fingerprints by fetching random product molecules 
    to modify a given, positive reaction SMILES
    For simplicity, only modifies product fingerprint, not the reactants
    
    TODO: Currently made for fingerprints - think about how to make this a base class 
    for both fingerprints and graphs. 
    '''
    def __init__(self, num_neg: int, lookup_dict: dict, 
                mol_fps: sparse_fp, rxn_type: str='diff', fp_type: str='count'):
        self.num_neg = num_neg
        self.lookup_dict = lookup_dict
        self.mol_fps = mol_fps
        self.rxn_type = rxn_type
        self.fp_type = fp_type
        self.orig_idxs = range(self.mol_fps.shape[0])
        # self.rdm_idxs = random.sample(self.orig_idxs, k=len(self.orig_idxs))
        # using this means that a product molecule (if it appears multiple times in different rxns)
        # will consistently map to a particular randomly selected molecule, which might not be a bad thing! 
        # we use the original prod_idx in lookup_dict to access self.rdm_idxs 
        # i.e. w/o this, a product molecule will stochastically map to different molecules throughout training

    def gen_one_sample(self, rxn_smi):
        ''' 
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            rdm_prod_idx = random.choice(self.orig_idxs)
            rdm_prod_fp = self.mol_fps[rdm_prod_idx]
            neg_rxn_fp = make_rxn_fp(rcts_fp, rdm_prod_fp, self.rxn_type)
            neg_rxn_fps.append(neg_rxn_fp) 
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)
        return neg_rxn_fps # torch.Tensor([pos_rxn_fp, *neg_rxn_fps])
        
class BitAugmentor:
    '''
    Generates negative reaction fingerprints by randomly switching the values of bits
    in a given positive reaction fingerprint
    For bit rxn fingerprints, this means randomly replacing values with 1, 0, or -1 
    For count rxn fingerprints, this means randomly adding 1 to the original value at a randomly chosen position

    Future strategies include:
        1) 'Attacks' the most sensitive bits by some analysis function 
            i.e. the bit that the model is currently most sensitive to
        2) Replace with a section of bits from a different molecule 
            (mimics RICAP from computer vision)
        3) Similar to 2, but select the 'donor' molecule using some similarity (or dissimilarity) metric
    NOTE: only compatible with fingerprints

    TODO: precompute rxn_fp & just load them
    TODO: of course, to precompute the entire set of pos_rxn_fp & neg_rxn_fp
    '''
    def __init__(self, num_neg: int, num_bits: int, strategy: str, 
                lookup_dict: dict, mol_fps: sparse_fp, 
                rxn_type: str='diff', fp_type: str='count'):
        self.num_neg = num_neg
        self.num_bits = num_bits
        self.strategy = strategy
        self.lookup_dict = lookup_dict
        self.mol_fps = mol_fps
        self.rxn_type = rxn_type
        self.fp_type = fp_type

    def gen_one_sample_count(self, rxn_smi):
        ''' For count fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(range(pos_rxn_fp.shape[-1]), k=self.num_bits) 
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = neg_rxn_fp[0, bit_idx] + 1 
            neg_rxn_fps.append(neg_rxn_fp) 
        return neg_rxn_fps # torch.Tensor([pos_rxn_fp, *neg_rxn_fps])
    
    def gen_one_sample_bit(self, rxn_smi):
        ''' For bit fingerprints
        Also see: rcts_prod_fps_from_rxn_smi, make_rxn_fp
        '''
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = []
        for i in range(self.num_neg):
            neg_rxn_fp = pos_rxn_fp.copy()
            rdm_bit_idxs = random.sample(range(pos_rxn_fp.shape[-1]), k=self.num_bits) 
            for bit_idx in rdm_bit_idxs:
                neg_rxn_fp[0, bit_idx] = random.choice([-1, 0, 1])
            neg_rxn_fps.append(neg_rxn_fp) 
        return neg_rxn_fps #torch.Tensor([pos_rxn_fp, *neg_rxn_fps])

class Augmentor:
    '''
    Parameters
    ----------
    augmentations : dict
        key : str 
            name of augmentation 
            choose from 'random', 'cosine', 'bit' 
        value : dict
            augmentation parameters, where
            key = name of parameter, value = value of that parameter
            e.g. augmentations['bit'] = {'num_neg': 5, 'num_bits': 5, 'strategy': 'default'}
            'random': 
                num_neg : int 
                    number of negative reactions to generate
            'cosine':
                num_neg : int
                    number of negative reactions to generate
                query_params : dict
                    num_threads : int (Default = 4)
                        number of CPU threads to use for kNN query by nmslib search index
                    efSearch : int (Default = 100)
                        100 is the recommended value to get high recall (96%)
                    **kwargs :
                        see nmslib's setQueryTimeParams documentation for other possible kwargs
            'bit': 
                num_neg : int
                    number of negative reactions to generate
                num_bits : int
                    number of bits to corrupt
                strategy : Optional[str]
                    the strategy to corrupt the bits. TODO: implemented soon!!!
    rxn_type : str (Default = 'diff')
        the method to calculate reaction fingerprints
        currently supports 'diff' & 'sep' methods
    fp_type : str (Default = 'count')
        the type of the fingerprints being supplied
        currently supports 'count' & 'bit' fingerprints
    root : str (Default = None)
        full path to the folder containing all the necessary files, which includes
        lookup_dict, sparse mol_fps, search_index. 
        If not provided, aka None, it defaults to full/path/to/rxn-ebm/data/cleaned_data/
    num_workers : int (Default = 0)
        how many workers to parallelize the PyTorch dataloader over
    '''
    def __init__(self, augmentations: dict, 
                lookup_dict_filename: str, mol_fps_filename: str, search_index_filename: str, 
                rxn_type: Optional[str]='diff', fp_type: Optional[str]='count',
                root: Optional[str]=None, num_workers: Optional[int]=0, **kwargs):
        self.lookup_dict_filename = lookup_dict_filename
        self.mol_fps_filename = mol_fps_filename
        self.search_index_filename = search_index_filename
        if root is None: # set root = rxn-ebm/
            root = Path(__file__).parents[1] / 'data' / 'cleaned_data'
        self.root = root
        with open(self.root / self.lookup_dict_filename, 'rb') as handle:
            self.lookup_dict = pickle.load(handle)
        self.mol_fps = sparse.load_npz(self.root / self.mol_fps_filename)
        
        self.rxn_smis = None
        self.fp_type = fp_type
        self.rxn_type = rxn_type
        self.num_workers = num_workers

        self.augs = ['cos', 'rdm', 'bit'] # full list of augs available
        for key, value in augmentations.items(): # augs to activate
            if key == 'cosine' or key == 'cos':
                self._init_cosine(**value) 
                self.augs.remove('cos') # find better way to append method than string
            elif key == 'random' or key == 'rdm':
                self._init_random(**value)
                self.augs.remove('rdm')
            elif key == 'bit' or key == 'bits':
                self._init_bit(**value)
                self.augs.remove('bit')
        for aug in self.augs: # augs to inactivate
            if aug == 'cos':
                self.cosaugmentor = None
            elif aug == 'rdm':
                self.rdmaugmentor = None
            elif aug == 'bit':
                self.bitaugmentor = None

    def _init_cosine(self, num_neg: int, query_params: Optional[dict]=None):
        ''' 
        NOTE: nmslib only accepts str for its filename, not Path objects 
        '''
        print('Initialising CosineAugmentor...')
        search_index = None
        if self.num_workers == 0: # else, load it from _init_worker_fn_
            search_index = nmslib.init(
                method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
            search_index.loadIndex(str(self.root / self.search_index_filename), load_data=True)
            if query_params:  
                search_index.setQueryTimeParams(query_params)
            else:
                search_index.setQueryTimeParams({'efSearch': 100})
        self.cosaugmentor = CosineAugmentor(
            num_neg, self.lookup_dict, self.mol_fps, search_index, self.rxn_type, self.fp_type)
    
    def _init_random(self, num_neg: int):
        print('Initialising RandomAugmentor...')
        self.rdmaugmentor = RandomAugmentor(
            num_neg, self.lookup_dict, self.mol_fps, self.rxn_type, self.fp_type)
        
    def _init_bit(self, num_neg: int, num_bits: int, strategy: Optional[str]=None):
        print('Initialising BitAugmentor...')
        self.bitaugmentor = BitAugmentor(
            num_neg, num_bits, strategy, self.lookup_dict, self.mol_fps, self.rxn_type, self.fp_type)

    def get_one_sample(self, rxn_smi: str) -> sparse_fp :
        ''' prepares one sample, which is 1 pos_rxn + K neg_rxns 
        where K is the sum of all of the num_neg for each active augmentation
        '''
        # only calls that augmentor if it exists
        # need to maintain an attribute self.augs 
        # that stores a list of augmentations to apply for every pos_rxn
        rcts_fp, prod_fp = rcts_prod_fps_from_rxn_smi(
            rxn_smi, self.fp_type, self.lookup_dict, self.mol_fps)  
        pos_rxn_fp = make_rxn_fp(rcts_fp, prod_fp, self.rxn_type)

        neg_rxn_fps = [] # for aug in self.augs:  # call each augmentor class 
        # prolly better way to execute than this... this seems quite inefficient
        # find out a way to only call those augs that are 'activated'
        if self.rdmaugmentor: 
            rdm_negs = self.rdmaugmentor.gen_one_sample(rxn_smi)
            neg_rxn_fps.extend(rdm_negs)

        if self.cosaugmentor:
            cos_negs = self.cosaugmentor.gen_one_sample(rxn_smi)
            neg_rxn_fps.extend(cos_negs)

        if self.bitaugmentor:
            if self.fp_type == 'bit':
                bit_negs = self.bitaugmentor.gen_one_sample_bit(rxn_smi)
            elif self.fp_type == 'count':
                bit_negs = self.bitaugmentor.gen_one_sample_count(rxn_smi)
            neg_rxn_fps.extend(bit_negs)

        out = sparse.hstack([pos_rxn_fp, *neg_rxn_fps])
        return out
        # spy_sparse2torch_sparse(out) # torch.Tensor([pos_rxn_fp, *neg_rxn_fps])

    def __getitem__(self, idx: int) -> sparse_fp:
        ''' Called by ReactionDataset.__getitem__(idx)
        '''
        return self.get_one_sample(self.rxn_smis[idx])

    def precompute(self, output_filename: str, 
                rxn_smis: Optional[Union[List[str], str]]=None) -> sparse_fp:
        if rxn_smis:
            self.load_smis(rxn_smis)
        else:
            assert self.rxn_smis is not None, 'Augmentor has no rxn_smis to precompute!'

        out = []
        for i in tqdm(range(len(self.rxn_smis))):
            out.append(self.__getitem__(i))
        out = sparse.vstack(out)
        sparse.save_npz(self.root / output_filename, out)
        return out

    def load_smis(self, rxn_smis: Union[List[str], str]):
        if isinstance(rxn_smis, list) and isinstance(rxn_smis[0], str):
            print('List of reaction SMILES strings detected.')
            self.rxn_smis = rxn_smis
        elif isinstance(rxn_smis, str):
            print('Loading reaction SMILES from filename string provided.')
            with open(self.root / rxn_smis, 'rb') as handle:
                self.rxn_smis = pickle.load(handle)
        else:
            raise Exception('Error! No reaction SMILES provided.')
        self.shape = (len(self.rxn_smis), self.mol_fps[0].shape[-1]) 
        # e.g. (40004, 4096) for train, needed to allow .shape[0] attribute from ReactionDataset.__len__()

def spy_sparse2torch_sparse(data: scipy.sparse.csr_matrix) -> tensor:
    """
    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples = data.shape[0]
    features = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row,coo_data.col])
    tensor = torch.sparse.IntTensor(
                indices, torch.from_numpy(values), [samples, features])
    return tensor

if __name__ == '__main__':
    from tqdm import tqdm

    augmentations = {
        'rdm': {'num_neg': 2}, 
        'cos': {'num_neg': 2},
        'bit': {'num_neg': 2, 'num_bits': 3}
    }
    lookup_dict_filename = '50k_mol_smi_to_count_fp.pickle'
    mol_fps_filename = '50k_count_mol_fps.npz'
    search_index_filename = '50k_cosine_count.bin'
    augmentor = Augmentor(augmentations, lookup_dict_filename, mol_fps_filename, search_index_filename)

    augmentor.load_smis(rxn_smis='50k_clean_rxnsmi_noreagent_train.pickle')
    augmentor.precompute(output_filename='50k_rdm_5_train.npz')
    
    augmentor.load_smis(rxn_smis='50k_clean_rxnsmi_noreagent_valid.pickle')
    augmentor.precompute(output_filename='50k_rdm_5_valid.npz')

    augmentor.load_smis(rxn_smis='50k_clean_rxnsmi_noreagent_test.pickle')
    augmentor.precompute(output_filename='50k_rdm_5_test.npz')

    # with open('data/cleaned_data/50k_clean_rxnsmi_noreagent_train.pickle', 'rb') as handle:
    #     rxnsmi_train = pickle.load(handle)
    # samples = []
    # for i in tqdm(range(len(rxnsmi_train))):
    #     samples.append(augmentor.get_one_sample(rxnsmi_train[i])) 

##################################################################
# def mol_fps_to_rxn_fp(mol_fps: List[np.ndarray], fp_type: Optional[str] = 'diff'
#                      ) -> np.ndarray or Tuple[np.ndarray, np.ndarray]:
#     ''' TODO: adapt this for sparse fingerprints (or maybe don't need to change anything at all)
#     Parameters
#     ----------
#     mol_fps : List[np.ndarray]
#         list of np.ndarray fingerprints of all reactant molecules and the product molecule in a reaction
#         (current assumption: only 1 product molecule per reaction)

#     fp_type : Optional[str] (Default = 'diff')
#         'diff' (difference):
#         Creates difference reaction fingerprint 
#         rxn_fp = prod_fp - sum(rct_fps)
#         Reference: Schneider et al, J. Chem. Inf. Model. 2015, 55, 1, 39–53
        
#         'sep' (separate):
#         Creates separate sum(rcts_fp) and prod_fp, then concatenates them 
#         Reference: Gao et al, ACS Cent. Sci. 2018, 4, 11, 1465–1476
    
#     Returns
#     -------
#     rxn_fp : np.ndarray
#         The processed reaction fingerprint, ready to be augmented (e.g. bit corruption) 
#         or fed into the model 

#         NOTE: the length of a 'sep' fingerprint will be the sum of 
#         the length of a reactant fingerprint and the length of a product fingerprint
#     '''    
#     # access product fingerprint
#     if fp_type == 'diff':
#         diff_fp = mol_fps[-1]
#     elif fp_type == 'sep':
#         prod_fp = mol_fps[-1]
                                  
#     # access reactant fingerprints
#     for i in range(len(mol_fps[: -1])):
#         if fp_type == 'diff': # subtract each reactant fingerprint from the product fingerprint
#             diff_fp = diff_fp - mol_fps[i]
#         elif fp_type == 'sep': # sum the reactant fingerprints
#             rcts_fp = rcts_fp + mol_fps[i]
    
#     if fp_type == 'diff':
#         return diff_fp
#     elif fp_type == 'sep':
#         return np.concatenate([rcts_fp, prod_fp], axis=1) # can we avoid np.concatenate? 

# class Augmentor:
    ''' TODO: fix this 

    Augmentor prepares training samples of K reaction fingerprints: 
    [positive_rxn, negative_rxn_1, ..., negative_rxn_K-1] where K-1 = number of negative examples per reaction

    Parameters
    ----------
    trainargs : dict
        dictionary of hyperparameters for the current experiment

    dataset : str 
        whether we are using train, valid, or test dataset 
        used to append to all base_paths to get the full filepaths 
    
    base_path_rxnsmi : str
        base path to .pickle file containing reaction SMILES strings
        should have the form: 'path/to/file/prefix_of_rxnsmi_pickle' 
        e.g. 'USPTO_50k_data/clean_rxn_50k_nomap_noreagent' 
        internally, we append '_{dataset}.pickle' to this, e.g. '_train.pickle'

    augs : List[str]
        list of augmentations to apply 
        currently supported options: 'random', 'cosine', 'bit' 
    
    aug_params : dict # List[Tuple[int, int]]
        dictionary of parameters for the augmentations specified in augs, must be in the same order as augs 
        for random sampling: 
            rdm_neg_prod, rdm_neg_rct : #  Optional[Tuple[int, int]]
                rdm_neg_prod : Number of product molecules to randomly sample and replace
                rdm_neg_rct : Number of reactant molecules to randomly sample and replace

        for cosine sampling:
            cos_neg_prod, cos_neg_rct : #  Optional[Tuple[int, int]]
                cos_neg_prod : Number of nearest neighbour product molecules to sample and replace
                cos_neg_rct : Number ofnearest neighbour reactant molecules to sample and replace            

        for bit corruption:
            bit_neg_rxn, bit_neg_bits : # Optional[Tuple[int, int]]]
                bit_neg_rxn : Number of negative reaction fingerprints to generate
                bit_neg_bits : Number of bits to randomly increase or decrease by 1 in each reaction fingerprint 

    Additional parameters
    --------------------
    For random sampling: 
        path_rct_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of reactant molecules 
            should have the form: 'path/to/file/rctfps_filename.npz' 
        path_prod_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of product molecules 
            should have the form: 'path/to/file/prodfps_filename.npz'     
            
    For cosine sampling:
        path_rct_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of reactant molecules 
            should have the form: 'path/to/file/rctfps_filename.npz' 
        path_prod_fps: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of product molecules 
            should have the form: 'path/to/file/prodfps_filename.npz'     
        path_allmol_fps : Optional[str] (Default = None)

        base_path_nnindex: Optional[str] (Default = None)
            path to .bin file containing the nearest neighbour search index on all the molecules across train, valid and test
            should have the form: 'path/to/file/prefix_of_search_index' 
            e.g. 'USPTO_50k_data/clean_rxn_50k_sparse_FPs'
            internally, we append f'_{dataset}.npz' to this, e.g. '_train.npz'

    For bit corruption:
        base_path_rxnfp: Optional[str] (Default = None)
            path to .npz file containing sparse fingerprints of reactions
            should have the form 'path/to/file/prefix_of_rxnfp' 
            e.g. 'USPTO_50k_data/clean_rxn_50k_sparse_rxnFPs'
            internally, we append f'_{dataset}.npz' to this, e.g. '_train.npz'
    
    Also see: mol_fps_to_rxn_fp
    '''
    # def __init__(self, trainargs: dict, dataset: str, base_path_rxnsmi: str,
    #             augs = List[str], aug_params = dict, # List[Tuple[int, int]],
    #             path_rct_fps: Optional[str] = None, path_prod_fps: Optional[str] = None, path_allmol_fps: Optional[str] = None, 
    #             base_path_nnindex: Optional[str] = None, base_path_rxnfp: Optional[str] = None): 
    #     self.trainargs = trainargs
    #     self.fp_type = self.trainargs['fp_type'] # needed for mol_fps_to_rxn_fp
    #     self.dataset = dataset

    #     for aug in augs:
    #         if aug == 'bit':
    #             full_path_rxnfp = base_path_rxnfp + f'_{dataset}.npz'
    #             self.rxn_fps = sparse.load_npz(full_path_rxnfp)
    #             self.rxn_fps_length = self.rxn_fps.shape[0]
    #             self.bit_neg_rxn = aug_params['bit_neg_rxn']
    #             self.bit_neg_bits = aug_params['bit_neg_bits']
            
    #         elif aug == 'cosine':
    #             self.rct_fps = sparse.load_npz(path_rct_fps)
    #             self.prod_fps = sparse.load_npz(path_prod_fps)
    #             self.allmol_fps = sparse.load_npz(path_allmol_fps)
    #             full_path_nnindex = base_path_nnindex + f'_{dataset}.npz'
            
    #         # elif aug == 'random':

    #         self.fp_raw_num_rcts = sparse.load_npz(base_path + '_' + dataset + '.npz')
    #         self.fp_raw_num_rcts_length = self.fp_raw_num_rcts.shape[0]           
    #         # to speed up faster_vstack & np.reshape

##############################################################

 
