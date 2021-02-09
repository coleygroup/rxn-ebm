# # # clean GLN, save as noGT_{phase}
python3 gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file clean_gln_cano_200 \
    --proposed_smi_file_prefixes GLN_retrain_200topk_200maxk_200beam \
    --proposers GLN_retrain \
    --topks 200 \
    --maxks 200

# # # clean retrosim, save as noGT_{phase}
python3 gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file clean_retrosim_cano_200 \
    --proposed_smi_file_prefixes retrosim_200maxtest_200maxprec \
    --proposers retrosim \
    --topks 200 \
    --maxks 200

# # # clean retroxpert, save as noGT_{phase}
python3 gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file clean_retroxpert_cano_200 \
    --proposed_smi_file_prefixes retroxpert_200topk_200maxk_200beam \
    --proposers retroxpert \
    --topks 200 \
    --maxks 200
