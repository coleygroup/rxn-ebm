conda activate openretro

python3 preprocess.py \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_preprocess" \
  --train_file="./data/gln_schneider50k/clean_train.csv" \
  --val_file="./data/gln_schneider50k/clean_valid.csv" \
  --test_file="./data/gln_schneider50k/clean_test.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --num_cores=64

# python preprocess.py \
#   --model_name="gln" \
#   --data_name="schneider50k" \
#   --log_file="gln_preprocess" \
#   --train_file="./data/gln_schneider50k/raw/raw_train.csv" \
#   --val_file="./data/gln_schneider50k/raw/raw_val.csv" \
#   --test_file="./data/gln_schneider50k/raw/raw_test.csv" \
#   --processed_data_path="./data/gln_schneider50k/processed" \
#   --num_cores=8
