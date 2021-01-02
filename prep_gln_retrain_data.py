import csv
import pickle
from tqdm import tqdm
import time

from rxnebm.data.preprocess import canonicalize

def main(remove_mapping : bool = False):
    start = time.time()
    rxn_class = "UNK"
    for phase in ['train', 'valid', 'test']:
        with open(f'rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_allmapped_{phase}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)

        with open(f'rxnebm/data/cleaned_data/clean_{phase}.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            # header
            writer.writerow(['id', 'class', 'reactants>reagents>production'])
            
            for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                rxn_smi_canon, _, _ = canonicalize.canonicalize_rxn_smi(rxn_smi, remove_mapping=remove_mapping)
                writer.writerow([i, rxn_class, rxn_smi_canon])
            
    print(f'Finished all phases! Elapsed: {time.time() - start:.2f} secs')
    # very fast, ~60 sec for USPTO-50k

if __name__ == '__main__':
    main(remove_mapping=False)