# rename files from retroxpert training & inference to:
# "retroxpert_top200_max200_beam200_raw_{phase}.csv"

python3 gen_proposals/gen_retroxpert.py \
        --parallelize \
        --topk=200 \
        --maxk=200 \
        --beam_size=200 \
        --csv_prefix "retroxpert_top200_max200_beam200_raw"

python3 gen_proposals/gen_union_or_clean_proposals.py \
    --train \
    --valid \
    --test \
    --log_file clean_retroxpert_cano_200 \
    --proposed_smi_file_prefixes "retroxpert_200topk_200maxk_200beam" \
    --proposers retroxpert \
    --topks 200 \
    --maxks 200

# now run make_fp.sh and/or make_graphfeat.sh