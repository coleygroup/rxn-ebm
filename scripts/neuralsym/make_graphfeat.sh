# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm

sym_seed=$1
python trainEBM.py \
    --log_file SYM_${sym_seed}_50top200max_gengraphfeat.log \
    --expt_name SYM_${sym_seed}_50top200max_gengraphfeat \
    --proposals_csv_file_prefix "neuralsym_200topk_200maxk_noGT_${sym_seed}" \
    --cache_suffix '50top_200max_stereo' \
    --representation "graph" \
    --minibatch_size 50 \
    --minibatch_eval 200