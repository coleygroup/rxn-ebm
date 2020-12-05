# Proposer modules
This should work standalone. @Min Htoo: feel free to merge as appropriate.


## Environmental setup
    TORCH_VER=1.6.0
    CUDA_VER=cu102 # for cpu, CUDA_VER=cpu 
    CUDA_VER2=10.2 # identical to CUDA_VER, just formatted differently
    
    conda create -n rxn-ebm-proposer python=3.6 scipy typing
    conda activate rxn-ebm-proposer
    conda install pytorch=${TORCH_VER} torchvision cudatoolkit=${CUDA_VER2} torchtext -c pytorch
    # for cpuonly pytorch, uncomment below line & comment out the above line
    # conda install pytorch=${TORCH_VER} torchvision cpuonly torchtext -c pytorch
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
    
    # install tensorflow (GPU)
    pip install tensorflow-gpu==1.14 gast==0.2.2
    # for cpuonly tensorflow, uncomment below line & comment out the above line
    # pip install tensorflow==1.14 gast==0.2.2

Note: if you encounter 'str' object has no attribute 'decode' when loading model weights for MT_Karpov, 
the problem is due to your h5py version being too high for tensorflow. For us, this fixed the issue: 

    pip uninstall h5py 

And then:

    pip install h5py==2.10.0 

## Sample usage
#### GLN Proposer
From the project root

    python rxnebm/proposer/gln_proposer_test.py

If the above doesn't work, please try

    python -m rxnebm.proposer.gln_proposer_test

#### Molecular Transformer (Karpov) Proposer
From the project root

    python rxnebm/proposer/mt_karpov_proposer_test.py

If the above doesn't work, please try

    python -m rxnebm.proposer.mt_karpov_proposer_test