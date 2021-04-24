python3 trainEBM.py \
  --ddp \
  --nodes 1 \
  --gpus 4 \
  --nr 0 \
  --port 12999 \
  --model_name="GraphEBM_projBoth" \
  --encoder_rnn_type gru \
  --atom_pool_type attention \
  --mol_pool_type sum \
  --encoder_hidden_size 300 \
  --encoder_inner_hidden_size 320 \
  --encoder_depth 10 \
  --encoder_dropout 0.04 \
  --proj_hidden_sizes 256 200 \
  --proj_activation "PReLU" \
  --proj_dropout 0.05 \
  --out_hidden_sizes 600 300 \
  --out_activation "PReLU" \
  --out_dropout 0.05 \
  --rxn_smis_file_prefix="50k_clean_rxnsmi_noreagent" \
  --onthefly \
  --do_compute_graph_feat \
  --cache_suffix='100top_400max_stereo' \
  --do_finetune \
  --do_test \
  --do_get_energies_and_acc \
    --log_file=g2e_best/10x300_320_256x200_600x300_SIM_GLN_STEREO_1MPN_100top400max_2e4_fac60_pat2_cool0_stop5_80ep_4GPU_${SLURM_JOBID}.log \
	--expt_name=10x300_320_256x200_600x300_SIM_GLN_STEREO_1MPN_100top400max_2e4_fac60_pat2_cool0_stop5_80ep_4GPU \
    --proposals_csv_file_prefix="GLN_50top200max_SIM_50top200max_noGT" \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--batch_size=2 \
    --batch_size_eval=4 \
	--minibatch_size=100 \
	--minibatch_eval=400 \
    --grad_clip=30 \
    --lr_floor_stop_training \
    --lr_floor 1e-9 \
    --lr_cooldown=0 \
    --warmup_epochs 0 \
	--learning_rate=2e-4 \
    --lr_scheduler='ReduceLROnPlateau' \
    --lr_scheduler_factor=0.6 \
    --lr_scheduler_patience=2 \
	--optimizer="Adam" \
	--epochs=80 \
    --early_stop \
    --early_stop_patience=5 \
	--num_workers=0 \
	--checkpoint \
    --test_on_train