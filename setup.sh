TORCH_VER=1.6.0
CUDA_VER=cu102

conda create -n rxnebm python=3.6 tensorflow-gpu=1.14 tqdm pathlib typing scipy pandas joblib -y
conda activate rxnebm

conda install -y pytorch=${TORCH_VER} torchvision cudatoolkit=10.1 torchtext -c pytorch
conda install -y -c conda-forge rdkit

pip install --no-binary :all: nmslib
pip install crem gdown networkx gast==0.2.2

# install rdchiral (needed by retrosim)
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"

# install torch geometric
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+{CUDA_VER}.html
# pip install torch-geometric

# install gln as a package
cd ./rxnebm/proposer/GLN_original
pip install -e .
cd ../../..

# install tensorflow
# pip install tensorflow-gpu==1.14 gast==0.2.2
