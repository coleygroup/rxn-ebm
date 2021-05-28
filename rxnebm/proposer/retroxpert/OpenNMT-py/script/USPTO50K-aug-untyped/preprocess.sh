# dataset=USPTO50K-aug-untyped
seed=$1
dataset=USPTO50K-aug-untyped_${seed}
dataset_prefix=USPTO50K-aug-untyped

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

# don't need for proposing
# python3  $PWD/../../onmt/bin/preprocess.py \
#     -train_src $PWD/../../data/${dataset}/src-train.txt \
#     -train_tgt  $PWD/../../data/${dataset}/tgt-train.txt \
#     -save_data  $PWD/../../data/${dataset}/${dataset} \
#     --src_seq_length 1000 --tgt_seq_length 1000 \
#     --src_vocab_size 1000 --tgt_vocab_size 1000 --share_vocab  --overwrite

# original args for training
python3  $PWD/../../onmt/bin/preprocess.py \
    -train_src $PWD/../../data/${dataset}/src-train-aug-err.txt \
    -train_tgt  $PWD/../../data/${dataset}/tgt-train-aug-err.txt \
    --valid_src  $PWD/../../data/${dataset}/src-valid.txt \
    --valid_tgt  $PWD/../../data/${dataset}/tgt-valid.txt \
    -save_data  $PWD/../../data/${dataset}/${dataset_prefix} \
    --src_seq_length 1000 --tgt_seq_length 1000 \
    --src_vocab_size 1000 --tgt_vocab_size 1000 --share_vocab  --overwrite
