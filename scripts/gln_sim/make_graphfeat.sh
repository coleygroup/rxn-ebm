# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm_g2e

gln_seed=$1
python trainEBM.py \
    --log_file GLN_${gln_seed}_50top200max_SIM_50top200max_gengraphfeat.log \
	  --expt_name GLN_${gln_seed}_50top200max_SIM_50top200max_gengraphfeat \
    --proposals_csv_file_prefix "GLN_50topk_200maxk_${gln_seed}_retrosim_50topk_200maxk_noGT" \
    --cache_suffix '100top_400max_stereo' \
  	--representation "graph" \
	  --minibatch_size 100 \
	  --minibatch_eval 400

