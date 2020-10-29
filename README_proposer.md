# Proposer modules
This should work standalone. @Min Htoo: feel free to merge as appropriate.


## Environmental setup
    TORCH_VER=1.6.0
    CUDA_VER=cu102
    
    conda create -n rxn-ebm-proposer python=3.6 scipy
    conda activate rxn-ebm-proposer
    conda install pytorch=${TORCH_VER} torchvision cudatoolkit=10.2 torchtext -c pytorch
    conda install rdkit=2019.03.3.0 -c rdkit
    
    # install torch geometric
    pip install torch-scatter==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-sparse==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-cluster==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-spline-conv==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
    pip install torch-geometric
    
    # install gln as a package
    cd ./rxnebm/proposer/GLN_original
    pip install -e .

## Sample usage
#### GLN Proposer
From the project root

    python rxnebm/proposer/gln_proposer_test.py