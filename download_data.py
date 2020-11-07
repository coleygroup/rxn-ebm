import gdown
import os

urls = ["https://drive.google.com/uc?id=1tyXIa_f20jzA8J5pvudSdwZ-t3dRnP8V",
        "https://drive.google.com/uc?id=1hSaXJB97YypV7Pav1qB0ovh1_oQLZs3t",
        "https://drive.google.com/uc?id=1XesKizw5E5IBXTTcIVNenz1H8QD69uyn",
        "https://drive.google.com/uc?id=12ZQPPYdugx7WDKjuXnSyHg6f6ILsn5sx",
        "https://drive.google.com/uc?id=1BLvjp5LjlPJg8W9KvJ3pcXiEE5alWL0M",
        "https://drive.google.com/uc?id=1iGrqy99TNBrHzRmLbSdchjgQ0yBnS2S7",
        "https://drive.google.com/uc?id=1rZoCn70np-5dfRown0wtnM54Iq-XZ3xk"]
output = "./rxnebm/data/cleaned_data/"

for url in urls:
    gdown.download(url, output, quiet=False)

fl = ["50k_mol_smis.pickle",
      "50k_mol_smi_to_sparse_fp_idx.pickle",
      "50k_sparse_fp_idx_to_mol_smi.pickle",
      "50k_count_mol_fps.npz",
      "50k_cosine_count.bin",
      "50k_cosine_count.bin.dat",
      "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"]

for fn in fl:
    assert os.path.exists(os.path.join(output, fn))
