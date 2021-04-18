# union of GLN + SIM, 50+50, 200+200
python3 gen_proposals/gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file gln_50_retrosim_50_cano \
    --proposed_smi_file_prefixes GLN_retrain_200topk_200maxk_noGT,retrosim_200topk_200maxk_noGT \
    --proposers GLN_retrain,retrosim \
    --topks 50,50 \
    --maxks 200,200

# just for double-checking purposes
python3 gen_proposals/gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file gln_50_retrosim_50_cano_doublecheck \
    --proposed_smi_file_prefixes GLN_retrain_50topk_200maxk_retrosim_50topk_200maxk_noGT \
    --proposers union \
    --topks 100 \
    --maxks 400