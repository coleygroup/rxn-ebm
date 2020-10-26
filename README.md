# rxn-ebm
Energy-based modeling of chemical reactions

## Environmental setup
#### Using Conda
    conda create -n rxnebm python=3.8 tqdm pathlib typing scipy
    conda activate rxnebm
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
    # conda install pytorch torchvision cpuonly -c pytorch # to install cpuonly build of pytorch
    
    # install latest version of rdkit 
    conda install -c conda-forge rdkit 
    
    # install nmslib
    pip install nmslib
    
    # install crem
    pip install crem

## Data setup
The data was obtained from https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1
We rename these 3 excel files to 'schneider50k_train.csv', 'schneider50k_test.csv' and 'schneider50k_valid.csv', and save them to data/original_data
Then, simply adjust the parameters as you wish in trainEBM.py and run the script. (Currently adding more arguments to be parsed from command-line)  

## Folder organisation
```
 rxn-ebm
    ├── trainEBM.py
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
    ├── checkpoints
    ├── scores
    └── notebooks
         └── data_exploration.ipynb 
 ```
