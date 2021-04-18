# retroxpert
python3 gen_proposals/gen_fps_from_proposals.py \
  --proposer="retroxpert" \
  --rxn_smi_file_prefix="50k_clean_rxnsmi_noreagent_canon" \
    --log_file=retroxpert_50top200max_makefp_${SLURM_JOBID}.log \
    --output_file_prefix="retroxpert_rxn_fps_50topk_200maxk_16384_hybrid_all" \
    --topk 50 \
    --maxk 200 \
    --representation "fingerprints" \
    --radius 3 \
    --fp_size 16384 \
    --rxn_type "hybrid_all" \
    --fp_type "count"

# example hybrid of GLN + retrosim, 50+50, 200+200
# python3 gen_proposals/gen_fps_from_proposals.py \
#   --proposer="union" \
#   --rxn_smi_file_prefix="50k_clean_rxnsmi_noreagent_canon" \
#     --log_file=gln_retrain_50top200max_sim_50top200max_makefp_${SLURM_JOBID}.log \
#     --proposals_file_prefix="GLN_retrain_50topk_200maxk_retrosim_50topk_200maxk_noGT" \
#     --output_file_prefix="GLN_retrain_retrosim_rxn_fps_100topk_400maxk_16384_hybrid_all" \
#     --topk 100 \
#     --maxk 400 \
#     --representation "fingerprints" \
#     --radius 3 \
#     --fp_size 16384 \
#     --rxn_type "hybrid_all" \
#     --fp_type "count"