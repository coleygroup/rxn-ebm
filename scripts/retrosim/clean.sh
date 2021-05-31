# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm

python gen_proposals/gen_union_or_clean_proposals.py \
    --log_file "clean_retrosim" \
    --proposed_smi_file_prefixes "retrosim_200maxtest_200maxprec" \
    --output_file_prefix "retrosim_200topk_200maxk_noGT" \
    --proposers "retrosim" \
    --topks 200 \
    --maxks 200