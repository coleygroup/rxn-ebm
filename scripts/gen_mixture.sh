python3 gen_mixture.py \
    --proposed_smi_file_prefixes GLN_retrain_200topk_200maxk_200beam retrosim_200maxtest_200maxprec retroxpert_200topk_200maxk_200beam \
    --output_file_prefix GLN_retrain_200topk_retrosim_200topk_retroxpert_200topk \
    --radius 3 \
    --fp_size 20480 \
    --phases train valid test

    # --parallelize \