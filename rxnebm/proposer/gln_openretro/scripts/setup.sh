TORCH_VER=1.6.0
CUDA_VER=10.1
CUDA_CODE=cu101

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda create -y -n openretro_cu101 python=3.6 tqdm
conda activate openretro_cu101

conda install -y pytorch=${TORCH_VER} torchvision torchaudio cudatoolkit=${CUDA_VER} -c pytorch
conda install -y rdkit -c rdkit

# install PTG
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_CODE}.html
pip install torch-geometric

# install opennmt, only needed for transformer
# pip install OpenNMT-py==1.2.0

# GLN installation, make sure to install on a machine with cuda
module load cuda/10.1
module load gcc/8.3.0

cd ./models/gln_model
pip install -e .
cd ../..