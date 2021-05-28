# dataset=USPTO50K-aug-untyped
seed=$1
dataset=USPTO50K-aug-untyped_${seed}
dataset_prefix=USPTO50K-aug-untyped

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

# just score valid, added seed
python3  ../../score_predictions.py  --beam_size 200 --invalid_smiles \
    --predictions  ../../experiments/results_${seed}/predictions_on_${dataset_prefix}_valid-prediction_top200.txt \
    --targets  ../../data/${dataset}/tgt-valid.txt \
    --sources  ../../data/${dataset}/src-valid.txt
