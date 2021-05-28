# dataset=USPTO50K-aug-untyped
seed=$1
dataset=USPTO50K-aug-untyped_${seed}
dataset_prefix=USPTO50K-aug-untyped
suffix=test-prediction

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

# added seed
python3  ../../score_predictions.py  --beam_size 50 --invalid_smiles \
    --predictions  ../../experiments/results_${seed}/predictions_on_${dataset_prefix}_${suffix}.txt \
    --targets  ../../data/${dataset}/tgt-test.txt \
    --sources  ../../data/${dataset}/src-${suffix}.txt

# original from repo
# python3  ../../score_predictions.py  --beam_size 50 --invalid_smiles \
#     --predictions  ../../experiments/results/predictions_on_${dataset}_${suffix}.txt \
#     --targets  ../../data/${dataset}/tgt-test.txt \
#     --sources  ../../data/${dataset}/src-${suffix}.txt

# new args for proposing top200
# python3  ../../score_predictions.py  --beam_size 200 --invalid_smiles \
#     --predictions  ../../experiments/results/predictions_on_${dataset}_train-prediction_top200.txt \
#     --targets  ../../data/${dataset}/tgt-train.txt \
#     --sources  ../../data/${dataset}/src-train.txt

# python3  ../../score_predictions.py  --beam_size 200 --invalid_smiles \
#     --predictions  ../../experiments/results/predictions_on_${dataset}_valid-prediction_top200.txt \
#     --targets  ../../data/${dataset}/tgt-valid.txt \
#     --sources  ../../data/${dataset}/src-valid.txt

# python3  ../../score_predictions.py  --beam_size 200 --invalid_smiles \
#     --predictions  ../../experiments/results/predictions_on_${dataset}_test-prediction_top200.txt \
#     --targets  ../../data/${dataset}/tgt-test.txt \
#     --sources  ../../data/${dataset}/src-test.txt

# old
# python3  ../../score_predictions.py  --beam_size 50 --invalid_smiles \
#     --predictions  ../../experiments/results/predictions_on_${dataset}_valid-prediction.txt \
#     --targets  ../../data/${dataset}/tgt-valid.txt \
#     --sources  ../../data/${dataset}/src-valid.txt

# original args for testing
# python3  ../../score_predictions.py  --beam_size 50 --invalid_smiles \
#     --predictions  ../../experiments/results/predictions_on_${dataset}_${suffix}.txt \
#     --targets  ../../data/${dataset}/tgt-test.txt \
#     --sources  ../../data/${dataset}/src-${suffix}.txt