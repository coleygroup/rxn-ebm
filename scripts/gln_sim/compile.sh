# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm

gln_seed=$1
python gen_proposals/gen_union_or_clean_proposals.py \
    --log_file gln_50_${gln_seed}_retrosim_50 \
    --proposed_smi_file_prefixes GLN_200topk_200maxk_noGT_${gln_seed},retrosim_200topk_200maxk_noGT \
    --proposers GLN,retrosim \
    --output_file_prefix GLN_50topk_200maxk_${gln_seed}_retrosim_50topk_200maxk_raw \
    --topks 50,50 \
    --maxks 200,200

# needed to ensure totally no ground-truth in the training negative proposals
python gen_proposals/gen_union_or_clean_proposals.py \
    --log_file gln_50_${gln_seed}_retrosim_50_doublecheck \
    --proposed_smi_file_prefixes GLN_50topk_200maxk_${gln_seed}_retrosim_50topk_200maxk_raw \
    --proposers union \
    --output_file_prefix GLN_50topk_200maxk_${gln_seed}_retrosim_50topk_200maxk_noGT \
    --topks 100 \
    --maxks 400