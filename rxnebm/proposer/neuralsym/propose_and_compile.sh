# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate neuralsym

sym_seed=$1
{
    python infer_all.py \
        --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
        --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
        --templates_file 50k_training_templates \
        --rxn_smi_prefix 50k_clean_rxnsmi_noreagent_allmapped_canon \
        --log_file "infer_${sym_seed}_highway_depth0_dim300" \
        --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
        --hidden_size 300 \
        --depth 0 \
        --topk 200 \
        --maxk 200 \
        --model Highway \
        --expt_name "Highway_${sym_seed}_depth0_dim300_lr1e3_stop2_fac30_pat1" \
        --seed ${sym_seed} \
        --print_accs
} && {
    mv data/neuralsym_200topk_200maxk_noGT_${sym_seed}_train.csv ../../data/cleaned_data/
    mv data/neuralsym_200topk_200maxk_noGT_${sym_seed}_valid.csv ../../data/cleaned_data/
    mv data/neuralsym_200topk_200maxk_noGT_${sym_seed}_test.csv ../../data/cleaned_data/
    printf "successfully proposed & compiled neuralsym for seed ${sym_seed}!"
}

