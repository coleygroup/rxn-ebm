# add these for training on multiple GPUs - just type a random port number between 0 and 65535
#   --ddp \
#   --nodes 1 \
#   --gpus 4 \
#   --nr 0 \
#   --port 1119 \

# these are the best hyperparams so far for GLN_retrain
python3 trainEBM.py \
  --model_name="GraphEBM_sep_projBoth_FFout" \
  --encoder_rnn_type gru \
  --atom_pool_type attention \
  --mol_pool_type sum \
  --encoder_hidden_size 300 \
  --encoder_inner_hidden_size 320 \
  --encoder_depth 10 \
  --encoder_dropout 0.075 \
  --encoder_activation "ReLU" \
  --proj_hidden_sizes 256 200 \
  --proj_activation "PReLU" \
  --proj_dropout 0.2 \
  --out_hidden_sizes 600 300 \
  --out_activation "PReLU" \
  --out_dropout 0.2 \
  --rxn_smis_file_prefix="50k_clean_rxnsmi_noreagent" \
  --onthefly \
  --do_compute_graph_feat \
  --cache_suffix='50top_200max' \
  --do_finetune \
  --do_test \
  --do_get_energies_and_acc \
	--log_file=g2e_retroxpert/10x300_320_256x200_600x300_retroxpert_50top200max_2e4_fac60_pat3_stop7_80ep.log \
	--expt_name=10x300_320_256x200_600x300_retroxpert_50top200max_2e4_fac60_pat3_stop7_80ep \
    --proposals_csv_file_prefix="retroxpert_200topk_200maxk_200beam" \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--batch_size=2 \
    --batch_size_eval=4 \
	--minibatch_size=50 \
	--minibatch_eval=200 \
    --grad_clip=10 \
    --lr_floor_stop_training \
    --lr_floor 1e-8 \
    --lr_cooldown=1 \
	--learning_rate=2e-4 \
    --lr_scheduler='ReduceLROnPlateau' \
    --lr_scheduler_factor=0.6 \
    --lr_scheduler_patience=3 \
	--optimizer="Adam" \
	--epochs=60 \
    --early_stop \
    --early_stop_patience=7 \
	--checkpoint \
    --test_on_train