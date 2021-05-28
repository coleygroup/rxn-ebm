module load gcc/8.3.0
module load cuda/10.1

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh

conda activate openretro_cu101

# second seed
python3 train.py \
  --do_train \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_train_vanilla" \
  --train_file="./data/gln_schneider50k/clean_train.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --model_path="./checkpoints/gln_schneider50k_20210423" \
  --model_seed=20210423

# third seed
python3 train.py \
  --do_train \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_train_vanilla" \
  --train_file="./data/gln_schneider50k/clean_train.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --model_path="./checkpoints/gln_schneider50k_77777777" \
  --model_seed=77777777

# first seed
python3 train.py \
  --do_train \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_train_vanilla" \
  --train_file="./data/gln_schneider50k/clean_train.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --model_path="./checkpoints/gln_schneider50k_19260817"
