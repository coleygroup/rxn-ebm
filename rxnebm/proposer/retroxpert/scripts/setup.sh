TORCH_VER=1.2.0
CUDA_VER=10.1
CUDA_CODE=cu101

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

conda install -y pytorch=${TORCH_VER} torchvision cudatoolkit=${CUDA_VER} torchtext -c pytorch
conda install -y rdkit=2019.03.4.0 -c rdkit

pip install dgl==0.4.2
pip install OpenNMT-py==1.0.0 networkx==2.4
