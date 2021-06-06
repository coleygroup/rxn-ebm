# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate neuralsym

# cannot use too high LR, will diverge slowly (loss increases > 20)
# higher bs --> faster training (using CPU)
# 8 sec/epoch on 1 RTX2080Ti
sym_seed=$1
python train.py \
    --model 'Highway' \
    --expt_name "Highway_${sym_seed}_depth0_dim300_lr1e3_stop2_fac30_pat1" \
    --log_file "Highway_${sym_seed}_depth0_dim300_lr1e3_stop2_fac30_pat1" \
    --do_train \
    --do_test \
    --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
    --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
    --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
    --bs 300 \
    --bs_eval 300 \
    --random_seed ${sym_seed} \
    --learning_rate 1e-3 \
    --epochs 30 \
    --early_stop \
    --early_stop_patience 2 \
    --depth 0 \
    --hidden_size 300 \
    --lr_scheduler_factor 0.3 \
    --lr_scheduler_patience 1 \
    --checkpoint