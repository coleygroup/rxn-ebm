import gdown
import os

urls_fns = [
    # for generating negative examples
    ("https://drive.google.com/uc?id=1zIN0T0tG-F1QDwb6QM7EJpUMl1Bw899f",
     "50k_clean_rxnsmi_noreagent_train.pickle"),
    ("https://drive.google.com/uc?id=1fxKp92MzsTha4Gd2wt721ZSakylKy70o",
     "50k_clean_rxnsmi_noreagent_valid.pickle"),
    ("https://drive.google.com/uc?id=1jcVgfSP_kG7DrNjyIJNOcH_MPkjWtAJR",
     "50k_clean_rxnsmi_noreagent_test.pickle"),
    ("https://drive.google.com/uc?id=1tyXIa_f20jzA8J5pvudSdwZ-t3dRnP8V",
     "50k_mol_smis.pickle"),
    # ("https://drive.google.com/uc?id=1hSaXJB97YypV7Pav1qB0ovh1_oQLZs3t",
    #  "50k_mol_smi_to_sparse_fp_idx.pickle"),
    # ("https://drive.google.com/uc?id=1XesKizw5E5IBXTTcIVNenz1H8QD69uyn",
    #  "50k_sparse_fp_idx_to_mol_smi.pickle"),
    # ("https://drive.google.com/uc?id=12ZQPPYdugx7WDKjuXnSyHg6f6ILsn5sx",
    #  "50k_count_mol_fps.npz"),
    # ("https://drive.google.com/uc?id=1BLvjp5LjlPJg8W9KvJ3pcXiEE5alWL0M",
    #  "50k_cosine_count.bin"),
    # ("https://drive.google.com/uc?id=1iGrqy99TNBrHzRmLbSdchjgQ0yBnS2S7",
    #  "50k_cosine_count.bin.dat"),
    ("https://drive.google.com/uc?id=1rZoCn70np-5dfRown0wtnM54Iq-XZ3xk",
     "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"),
    # pre-computed augmented data, for FeedforwardEBM (obsolete for now)
    # ("https://drive.google.com/uc?id=1kAuwfGv0s1OOo9be0NyNNhOekdWCwGLT",
    #  "50k_rdm_5_cos_5_bit_5_1_1_mut_10_train.npz"),
    # ("https://drive.google.com/uc?id=1BhcIeVsSSmRXpfCfTqsorUXWg_Tw5i7a",
    #  "50k_rdm_5_cos_5_bit_5_1_1_mut_10_valid.npz"),
    # ("https://drive.google.com/uc?id=13DwNxixNp_ylOTuA047mZSTgTCKL9WYm",
    #  "50k_rdm_5_cos_5_bit_5_1_1_mut_10_test.npz"),
    # retrosim CSV files
    ("https://drive.google.com/uc?id=1OaHvZS85yxhbEfFzvfrVe-uVxGC0NoKL",
     "retrosim_200topk_200maxk_noGT_train.csv"),
    ("https://drive.google.com/uc?id=1XjCoZ7N7q-eI6MJPPk6w7FT9-5FvuJTL",
     "retrosim_200topk_200maxk_noGT_valid.csv"),
    ("https://drive.google.com/uc?id=1eUpPnTwwe3X9tinpx5xjvLgXAzPBTjcQ",
     "retrosim_200topk_200maxk_noGT_test.csv"),
    # GLN CSV files
    ("https://drive.google.com/uc?id=15h4M0ZJDbSn9A3n9EuIoCHH2XfWrxi8P",
    "GLN_200topk_200maxk_noGT_train.csv"),
    ("https://drive.google.com/uc?id=1_KLldNX6Re6UuT68-IJleBXrpKYjf5q_",
    "GLN_200topk_200maxk_noGT_valid.csv"),
    ("https://drive.google.com/uc?id=1N_OwH3JZDfmvEzj0wi3TqxfI__IC9Yix",
    "GLN_200topk_200maxk_noGT_test.csv"),
    # retroxpert CSV files
    ("https://drive.google.com/uc?id=1Z5QOMHrc1q6MiZw1ojqd7x7f2f0Ux41p",
     "retroxpert_200topk_200maxk_noGT_train.csv"),
    ("https://drive.google.com/uc?id=1R24cq3VTKCq9YA9j2u_99wkRziilcGSn",
     "retroxpert_200topk_200maxk_noGT_valid.csv"),
    ("https://drive.google.com/uc?id=1KXOTIbXFXdpSGSYUSAQ4NYyZdvBfo7kj",
     "retroxpert_200topk_200maxk_noGT_test.csv"),

]
output = "./rxnebm/data/cleaned_data/"

for url, fn in urls_fns:
    ofn = os.path.join(output, fn)
    if not os.path.exists(ofn):
        gdown.download(url, output, quiet=False)
        assert os.path.exists(ofn)
    else:
        print(f"{ofn} exists, skip downloading")
