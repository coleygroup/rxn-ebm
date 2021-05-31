import argparse
import csv
import pickle
from tqdm import tqdm
import time

from rxnebm.data.preprocess import canonicalize

parser = argparse.ArgumentParser()
parser.add_argument('--output_format',
                    type=str,
                    default='gln',
                    help='["gln", "retroxpert"]')
# TODO: load schneider50k csv directly & clean it here instead of loading cleaned .pickle file
args = parser.parse_args()

def prep_canon_gln():
    start = time.time()
    rxn_class = "UNK"
    for phase in ['train', 'valid', 'test']:
        with open(f'rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_allmapped_canon_{phase}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)

        with open(f'rxnebm/data/cleaned_data/clean_gln_{phase}.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            # header
            writer.writerow(['id', 'class', 'reactants>reagents>production'])
            
            for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                writer.writerow([i, rxn_class, rxn_smi])
            
    print(f'Finished all phases! Elapsed: {time.time() - start:.2f} secs')
    # very fast, ~60 sec for USPTO-50k

def prep_canon_retroxpert():
    start = time.time()
    rxn_class = "UNK"
    for phase in ['train', 'valid', 'test']:
        with open(f'rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_allmapped_canon_{phase}.pickle', 'rb') as handle:
            rxn_smis = pickle.load(handle)

        with open(f'rxnebm/data/cleaned_data/{phase}.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            # header
            writer.writerow(['class', 'id', 'rxn_smiles'])
            
            for i, rxn_smi in enumerate(tqdm(rxn_smis, desc=f'Writing rxn_smi in {phase}')):
                rxn_smi_canon = canonicalize.canonicalize_products(rxn_smi)
                writer.writerow([rxn_class, i, rxn_smi_canon])
            
    print(f'Finished all phases! Elapsed: {time.time() - start:.2f} secs')
    # very fast, ~60 sec for USPTO-50k

if __name__ == '__main__':
    print(args.output_format)

    if args.output_format == 'gln':
        prep_canon_gln()
    elif args.output_format == 'retroxpert':
        prep_canon_retroxpert()
    else:
        raise ValueError