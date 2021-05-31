# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm

# multi-GPU is unnecessary for FeedforwardEBM, and in fact slows it down greatly due to inter-GPU communication
# overhead as the model is very large
ebm_seed=$1
sym_seed=$2
python trainEBM.py \
  --model_name "FeedforwardEBM" \
  --encoder_hidden_size 1024 128 \
  --encoder_dropout 0.2 \
  --encoder_activation "PReLU" \
  --out_hidden_sizes 128 \
  --out_activation "PReLU" \
  --out_dropout 0.2 \
  --do_train \
  --do_test \
  --do_get_energies_and_acc \
	--log_file 1024x128_128_SYM_${ebm_seed}_50top200max_lr1e3_fac20_pat0_stop2_40ep.log \
	--expt_name 1024x128_128_SYM_${ebm_seed}_50top200max_lr1e3_fac20_pat0_stop2_40ep \
    --proposals_csv_file_prefix "neuralsym_200topk_200maxk_noGT_${sym_seed}" \
    --precomp_rxnfp_prefix "neuralsym_${sym_seed}_rxn_fps_50topk_200maxk_16384_hybrid_all" \
	--representation "fingerprint" \
	--random_seed ${ebm_seed} \
	--batch_size 96 \
    --batch_size_eval 96 \
    --lr_floor 8e-7 \
	--learning_rate 1e-3 \
    --lr_scheduler_factor 0.2 \
    --lr_scheduler_patience 0 \
	--epochs 40 \
    --early_stop \
    --early_stop_patience 2 \
    --test_on_train \
    --checkpoint