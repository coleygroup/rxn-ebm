python3 gen_mixture.py \
    --proposed_smi_file_prefixes GLN_retrain_200topk_200maxk_200beam retrosim_200maxtest_200maxprec retroxpert_200topk_200maxk_200beam \
    --output_file_prefix GLN_retrain_1topk_retrosim_1topk_retroxpert_1topk_16384 \
    --radius 3 \
    --fp_size 16384 \
    --phases train valid test

    # --parallelize \