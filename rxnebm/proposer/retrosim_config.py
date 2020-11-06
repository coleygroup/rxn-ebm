retrosim_config = {
    "topk": 200,
    "max_prec": 200,
    "similarity_type": 'Tanimoto',
    "fp_type": "Morgan2Feat",
    # try to use absolute path if possible
    "input_data_folder": "./rxnebm/data/cleaned_data",
    "input_data_file_prefix": "50k_clean_rxnsmi_noreagent_allmapped",  #50k_evencleaner
    "output_folder": None,
    "parallelize": False,
}
