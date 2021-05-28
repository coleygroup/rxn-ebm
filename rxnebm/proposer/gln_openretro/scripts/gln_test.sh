conda activate openretro_cu101

module load cuda/10.1

python3 test.py \
  --test_all_ckpts \
  --model_name="gln" \
  --data_name="schneider50k" \
  --log_file="gln_test_vanilla" \
  --train_file="./data/gln_schneider50k/clean_train.csv" \
  --val_file="./data/gln_schneider50k/clean_valid.csv" \
  --test_file="./data/gln_schneider50k/clean_test.csv" \
  --processed_data_path="./data/gln_schneider50k/processed" \
  --model_path="./checkpoints/gln_schneider50k" \
  --test_output_path="./results/gln_schneider50k"
