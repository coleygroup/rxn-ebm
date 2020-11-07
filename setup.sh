TORCH_VER=1.6.0
CUDA_VER=cu102

conda create -n rxnebm python=3.6 tqdm pathlib typing scipy pandas joblib -y
conda activate rxnebm

conda install -y pytorch=${TORCH_VER} torchvision cudatoolkit=10.2 torchtext -c pytorch

# install latest version of rdkit
conda install -y -c conda-forge rdkit

# install nmslib
pip install nmslib

# install crem
pip install crem

# install rdchiral (needed by retrosim)
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"

# install torch geometric
pip install torch-scatter==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
pip install torch-sparse==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
pip install torch-cluster==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
pip install torch-spline-conv==latest+${CUDA_VER} -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}.html
pip install torch-geometric

# install gln as a package
cd ./rxnebm/proposer/GLN_original
pip install -e .
cd ../../..

# install tensorflow
pip install tensorflow-gpu==1.14 gast==0.2.2