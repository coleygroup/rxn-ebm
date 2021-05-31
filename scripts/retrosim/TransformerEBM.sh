# ensure conda is properly initialised, such as using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm

# for multi-GPU, choose any number (0-65535) for port
ebm_seed=$1
python trainEBM.py \
  --ddp \
  --nodes 1 \
  --gpus 4 \
  --nr 0 \
  --port 42213 \
  --model_name "TransformerEBM" \
  --do_train \
  --do_test \
  --do_get_energies_and_acc \
	--log_file 3x256_4h_512f_256e_CLS_256seq_SIM_${ebm_seed}_50top200max_bs8_lr2e3w0_fac60_pat2_stop6_100ep_4GPU.log \
	--expt_name 3x256_4h_512f_256e_CLS_256seq_SIM_${ebm_seed}_50top200max_bs8_lr2e3w0_fac60_pat2_stop6_100ep_4GPU \
    --vocab_file "vocab.txt" \
    --proposals_csv_file_prefix "retrosim_200topk_200maxk_noGT" \
	--representation "smiles" \
    --max_seq_len 256 \
    --encoder_embed_size 256 \
    --encoder_depth 3 \
    --encoder_hidden_size 256 \
    --encoder_num_heads 4 \
    --encoder_filter_size 512 \
    --encoder_dropout 0.05 \
    --attention_dropout 0.025 \
    --s2e_pool_type 'CLS' \
	--random_seed ${ebm_seed} \
	--batch_size 8 \
    --batch_size_eval 8 \
	--minibatch_size 50 \
	--minibatch_eval 200 \
    --lr_floor 1e-8 \
	--learning_rate 2e-3 \
    --lr_scheduler_factor 0.6 \
    --lr_scheduler_patience 2 \
	--epochs 100 \
    --early_stop \
    --early_stop_patience 5 \
	--checkpoint \
    --test_on_train
