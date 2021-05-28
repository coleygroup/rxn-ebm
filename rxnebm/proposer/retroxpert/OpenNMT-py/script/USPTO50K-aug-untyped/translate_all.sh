# dataset=USPTO50K-aug-untyped
seed=$1
dataset=USPTO50K-aug-untyped_${seed}
dataset_prefix=USPTO50K-aug-untyped
suffix=test-prediction

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate retroxpert_

# propose 200, with seed
python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
    --gpu 0  --src ../../data/${dataset}/src-valid-prediction.txt \
    --output ../../experiments/results_${seed}/predictions_on_${dataset_prefix}_valid-prediction_top200.txt \
    --beam_size 200  --n_best 200 \
    --batch_size 2 --replace_unk --max_length 300
    
python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
    --gpu 0  --src ../../data/${dataset}/src-test-prediction.txt \
    --output ../../experiments/results_${seed}/predictions_on_${dataset_prefix}_test-prediction_top200.txt \
    --beam_size 200  --n_best 200 \
    --batch_size 2 --replace_unk --max_length 300

python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
    --gpu 0  --src ../../data/${dataset}/src-train-prediction.txt \
    --output ../../experiments/results_${seed}/predictions_on_${dataset_prefix}_train-prediction_top200.txt \
    --beam_size 200  --n_best 200 \
    --batch_size 2 --replace_unk --max_length 300


# propose 200, without seed
# python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
#     --gpu 0  --src ../../data/${dataset}/src-valid-prediction.txt \
#     --output ../../experiments/results/predictions_on_${dataset}_valid-prediction_top200.txt \
#     --beam_size 200  --n_best 200 \
#     --batch_size 2 --replace_unk --max_length 300
    
# python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
#     --gpu 0  --src ../../data/${dataset}/src-test-prediction.txt \
#     --output ../../experiments/results/predictions_on_${dataset}_test-prediction_top200.txt \
#     --beam_size 200  --n_best 200 \
#     --batch_size 2 --replace_unk --max_length 300

# python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
#     --gpu 0  --src ../../data/${dataset}/src-train-prediction.txt \
#     --output ../../experiments/results/predictions_on_${dataset}_train-prediction_top200.txt \
#     --beam_size 200  --n_best 200 \
#     --batch_size 2 --replace_unk --max_length 300

# old
# python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
#     --gpu 0  --src ../../data/${dataset}/src-valid-prediction.txt \
#     --output ../../experiments/results/predictions_on_${dataset}_valid-prediction_top200.txt \
#     --beam_size 200  --n_best 200 \
#     --batch_size 2 --replace_unk --max_length 300
    
# python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
#     --gpu 0  --src ../../data/${dataset}/src-test-prediction.txt \
#     --output ../../experiments/results/predictions_on_${dataset}_test-prediction_top200.txt \
#     --beam_size 200  --n_best 200 \
#     --batch_size 2 --replace_unk --max_length 300

# python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
#     --gpu 0  --src ../../data/${dataset}/src-valid-prediction.txt \
#     --output ../../experiments/results/predictions_on_${dataset}_valid-prediction_with_scores.txt \
#     --beam_size 50  --n_best 50 \
#     --batch_size 8 --replace_unk --max_length 300

# python3 ../../onmt/bin/translate.py --model ../../experiments/checkpoints/${dataset}/average_model.pt \
#     --gpu 0  --src ../../data/${dataset}/src-${suffix}.txt \
#     --output ../../experiments/results/predictions_on_${dataset}_${suffix}.txt \
#     --beam_size 50  --n_best 50 \
#     --batch_size 8 --replace_unk --max_length 300