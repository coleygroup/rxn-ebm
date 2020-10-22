# rxn-ebm
Energy-based modeling of chemical reactions

# Environmental setup
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
    |    ├── crem_mutate.py
    |    ├── analyse_results.py
    |    ├── preprocess
    |    |        ├── clean_smiles.py    
    |    |        ├── smi_to_fp.py
    |    |        └── build_search_idx.py
    |    ├── original_data  
    |    └── cleaned_data
    ├── checkpoints
    ├── scores
    └── notebooks
         └── data_exploration.ipynb 
 ```
