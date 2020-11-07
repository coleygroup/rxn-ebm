# rxnebm
Energy-based modeling of chemical reactions

## Environmental setup
### Using Conda
    TORCH_VER=1.6.0
    CUDA_VER=cu102
    
    conda create -n rxnebm python=3.8 tqdm pathlib typing scipy pandas joblib 
    conda activate rxnebm
    
    conda install pytorch=${TORCH_VER} torchvision cudatoolkit=10.2 torchtext -c pytorch 
    # to install cpuonly build of pytorch
    # conda install pytorch=${TORCH_VER} torchvision cpuonly -c pytorch
    
    # install latest version of rdkit 
    conda install -c conda-forge rdkit 
    
    # install nmslib
    pip install nmslib
    
    # install crem
    pip install crem
    
    # install rdchiral (needed by retrosim) 
    pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
    
    # TODO: confirm with Zhengkai about package requirements from README_proposer.md 
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
    # install torch geometric
    pip install torch-scatter==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-sparse==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-cluster==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-spline-conv==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-geometric
    
    # install gln as a package
    cd ./rxnebm/proposer/GLN_original
    pip install -e .
    
   # install tensorflow # TODO: add cpu-only version 
    pip install tensorflow-gpu==1.14 gast==0.2.2


## Data setup
### Data source
The data was obtained from [the dropbox folder](https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1) provided by the authors of [GLN](https://github.com/Hanjun-Dai/GLN). 
We rename these 3 csv files from ```raw_{phase}.csv``` to ```'schneider50k_train.csv'```, ```'schneider50k_test.csv'``` and ```'schneider50k_valid.csv'```, and save them to ```rxnebm/data/original_data``` <br>

#### Data preprocessing
The entire data preprocessing pipeline can be run from ```trainEBM.prepare_data()```, which performs a series of data cleaning & preprocessing steps, in the following order:
1. Cleans the raw SMILES strings
2. Extracts all unique molecule SMILES strings as a list
3. Converts unique molecule SMILES into a matrix of Morgan count molecular fingerprints.
4. Generates 2 lookup tables (dictionaries), 1 to map molecular SMILES strings into the corresponding index in that matrix of Morgan count fingerprints, and the other to map the reverse: index (key) --> molecular SMILES (value) 
5. Builds a nearest-neighbour search index using the [nmslib](https://github.com/DrrDom/crem) package
6. Generates a (very) large database of [CReM](https://github.com/DrrDom/crem) negatives, mapping each product SMILES string in the dataset (key), to a list of highly similar, mutated product SMILES strings (value). Note that this step can take from 10-13 hours on the USPTO_50k dataset for 150 mutated products / original product, and that CReM does not guarantee the requested number of mutated products. To deal with this, we pad with vectors of all 0's, and implement a simple masking step in our network to ignore these vectors. <br>

### List of provided data for pre-training rxebm:
For ease of reproducibility, all data is [available on Google Drive](https://drive.google.com/drive/folders/1ISXFL7SuVY_sW3z36hQfpyDMH1KF22nS?usp=sharing):
- three ```50k_clean_rxnsmi_noreagent_{phase}.pickle``` files contain the cleaned reaction SMILES strings from USPTO_50k (generated by ```rxnebm/data/preprocess/clean_smiles.py```)
- ```50k_mol_smis.pickle``` (list of all unique molecule SMILES, generated by ```rxnebm/data/preprocess/clean_smiles.py```) 
- ```50k_mol_smi_to_sparse_fp_idx.pickle``` (the lookup table mapping molecular SMILES to Morgan count molecular fingerprints by their index in the sparse matrix ```50k_count_mol_fps.npz```, generated by ```rxnebm/data/preprocess/smi_to_fp.py```) 
- ```50k_sparse_fp_idx_to_mol_smi.pickle``` (the lookup table mapping Morgan count molecular fingerprints by their index to molecular SMILES, generated by ```rxnebm/data/preprocess/smi_to_fp.py```) 
- ```50k_count_mol_fps.npz``` (matrix of Morgan count moelcular fingerprints, generated by ```rxnebm/data/preprocess/smi_to_fp.py```) 
- ```50k_cosine_count.bin``` & ```50k_cosine_count.bin.dat``` (generated by ```rxnebm/data/preprocess/prep_nmslib.py```)
- ```50k_neg150_rad2_maxsize3_mutprodsmis.pickle``` (generated by ```rxnebm/data/preprocess/prep_crem.py```). ```50k_neg150_rad2_maxsize3_insufficient.pickle``` is a supplementary dictionary mapping each product SMILES (key) for which CReM could not generate the requested number of mutated molecules, to the number of negatives (value) that CReM could generate. <br>

### Extra-clean data for training retrosynthesis models from scratch:
For our fine-tuning step, we trained a zoo of popular & state-of-the-art retrosynthesis models. For all of these models, we use a single, extra-clean USPTO_50k dataset, split roughly into 80/10/10. These are derived from the three ``` schneider50k_{phase}.csv ``` files, using the script ```rxnebm/data/preprocess/clean_smiles.py```. This data is included in this repository under ```rxnebm/data/cleaned_data/``` as ```50k_clean_rxnsmi_noreagent_allmapped_{phase}.pickle```, and also in the [Google Drive](https://drive.google.com/drive/folders/1ISXFL7SuVY_sW3z36hQfpyDMH1KF22nS?usp=sharing) sub-folder ```Retro_Reproduction``` 
Specifically, we perform these steps:
1. Keep all atom mapping
2. Remove reaction SMILES strings with product molecules that are too small and clearly incorrect. The criteria used was ```len(prod_smi) < 3```. 4 reaction SMILES strings were caught by this criteria, with products: 		
    - ```'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccc([I:19])cc21>>[IH:19]'```
    - ```'O=C(CO)N1CCC(C(=O)[OH:28])CC1>>[OH2:28]'```
    - ```'CC(=O)[Br:4]>>[BrH:4]'```
    - ```'O=C(Cn1c(-c2ccc(Cl)c(Cl)c2)nc2cccnc21)[OH:10]>>[OH2:10]'```
3. Remove all duplicate reaction SMILES strings
4. Remove reaction SMILES in the training data that overlap with validation/test sets + validation data that overlap with the test set.
    - test_appears_in_train: 50
    - test_appears_in_valid: 6
    - valid_appears_in_train: 44
5. Finally, we obtain an (extra) clean dataset of reaction SMILES:
    - Train: 39713
    - Valid: 4989
    - Test: 5005
 
## Training
With the cleaned data generated by ```trainEBM.prepare_data()```, simply adjust the parameters as you wish in ```trainEBM.train()``` and run the script. This includes specifying the augmentations to apply, as well as model and training parameters. **(TODO) adding arguments to be parsed from command-line** <br> 

For additional convenience, a set of pre-computed reaction fingerprints has also been provided:
- ```50k_rdm_5_cos_5_bit_5_1_1_mut_10_{split}.npz```, the syntax being: ```{dataset_name}_rdm_{num_rdm_negs}_cos_{num_cos_negs}_bit_{num_bit_negs}_{num_bits}_{increment_bits}_mut_{num_mut_negs}_{split}.npz``` 

## Misc details
This project uses ``` black ``` for auto-formatting to the ``` pep8 ``` style guide, and ``` isort ``` to sort imports. ``` pylint ``` is also used in ``` vscode ``` to lint all code. 

## Folder organisation
```
 rxnebm
    ├── experiment
    │    ├── expt.py
    |    └── expt_utils.py
    ├── model
    |    ├── base.py
    │    ├── FF.py
    |    └── model_utils.py
    ├── data
    |    ├── dataset.py
    |    ├── augmentors.py
    |    ├── analyse_results.py
    |    ├── preprocess
    |    |        ├── clean_smiles.py    
    |    |        ├── smi_to_fp.py
    |    |        ├── prep_crem.py
    |    |        └── prep_nmslib.py
    |    ├── original_data  
    |    └── cleaned_data
    ├── proposer   
    |    ├── Retrosim_modified
    |    ├── retrosim_model.py
    |    ├── retrosim_proposer.py
    |    ├── GLN_original
    |    ├── gln_proposer.py
    |    ├── MT_karpov
    |    ├── mt_karpov_proposer.py
    ├── checkpoints
    └── scores
 ```
