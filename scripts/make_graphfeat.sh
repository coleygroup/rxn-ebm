    # NEURALSYM
    # --log_file=g2e_best/SYM_50top200max_gengraphfeat_${SLURM_JOBID}.log \
	# --expt_name=SYM_50top200max_gengraphfeat \
    # --proposals_csv_file_prefix="neuralsym_200topk_200maxk_noGT" \

    # RETROXPERT
    # --log_file=g2e_best/xpert_50top200max_gengraphfeat_${SLURM_JOBID}.log \
	# --expt_name=xpert_50top200max_gengraphfeat \
    # --proposals_csv_file_prefix="retroxpert_200topk_200maxk_noGT" \

    # RETROSIM
    # --log_file=g2e_best/sim_50top200max_gengraphfeat_${SLURM_JOBID}.log \
	# --expt_name=sim_50top200max_gengraphfeat \
    # --proposals_csv_file_prefix="retrosim_200topk_200maxk_noGT" \

    # GLN_retrain
    # --log_file=g2e_best/gln_retrain_50top200max_gengraphfeat_${SLURM_JOBID}.log \
	# --expt_name=gln_retrain_50top200max_gengraphfeat \
    # --proposals_csv_file_prefix="GLN_retrain_200topk_200maxk_noGT" \
python3 trainEBM.py \
  --model_name="GraphEBM_sep_projBoth_FFout" \
  --rxn_smis_file_prefix="50k_clean_rxnsmi_noreagent" \
  --onthefly \
  --do_finetune \
  --do_compute_graph_feat \
    --log_file=g2e_best/xpert_50top200max_gengraphfeat_${SLURM_JOBID}.log \
	--expt_name=xpert_50top200max_gengraphfeat \
    --proposals_csv_file_prefix="retroxpert_200topk_200maxk_noGT" \
    --cache_suffix='50top_200max_stereo' \
	--representation="smiles" \
	--random_seed=0 \
    --epochs=0 \
	--minibatch_size=50 \
	--minibatch_eval=200