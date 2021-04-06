
python3 gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file gln_35_retrosim_30_retroxpert_35_cano \
    --proposed_smi_file_prefixes GLN_retrain_200topk_200maxk_noGT,retrosim_200topk_200maxk_noGT,retroxpert_200topk_200maxk_noGT \
    --proposers GLN_retrain,retrosim,retroxpert \
    --topks 35,30,35 \
    --maxks 125,100,75

# just for double-checking purposes
python3 gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file gln_35_retrosim_30_retroxpert_35_cano_doublecheck \
    --proposed_smi_file_prefixes GLN_retrain_35topk_125maxk_retrosim_30topk_100maxk_retroxpert_35topk_75maxk_noGT \
    --proposers union \
    --topks 100 \
    --maxks 300