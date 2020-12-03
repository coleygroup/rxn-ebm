# Proposer modules
This should work standalone. @Min Htoo: feel free to merge as appropriate.


## Environmental setup
    TORCH_VER=1.6.0
    CUDA_VER=cu102
    CUDA_VER2=10.2 # identical to CUDA_VER, just formatted differently 
    
    conda create -n rxn-ebm-proposer python=3.6 scipy typing
    conda activate rxn-ebm-proposer
    conda install pytorch=${TORCH_VER} torchvision cudatoolkit=${CUDA_VER2} torchtext -c pytorch
    conda install -y -c conda-forge rdkit 
    
    # install torch geometric
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
    pip install torch-geometric
    
    # install gln as a package
    cd ./rxnebm/proposer/GLN_original
    pip install -e .
    
    # install tensorflow
    pip install tensorflow-gpu==1.14 gast==0.2.2

## Sample usage
#### GLN Proposer
From the project root

    python rxnebm/proposer/gln_proposer_test.py

#### Molecular Transformer (Karpov) Proposer
From the project root

    python rxnebm/proposer/mt_karpov_proposer_test.py
