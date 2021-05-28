# openretro
An open source library for retrosynthesis benchmarking.

# Environment setup
### Using conda
Assuming conda is installed and initiated (i.e. conda activate is a warning-free command), then

    bash -i scripts/setup.sh
    conda activate openretro

### Using docker (TODO)

# Models
## GLN
Adapted from original GLN (https://github.com/Hanjun-Dai/GLN)

Step 1. Prepare the raw atom-mapped .csv files for train, validation and test.
See https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1
for sample data format.

Step 2. Preprocessing: modify the args in scripts/gln_preprocess.sh, then

    sh scripts/gln_preprocess.sh

Step 3. Training: modify the args in scripts/gln_train.sh, then
    
    sh scripts/gln_train.sh

Step 4 (optional). Testing: modify the args in scripts/gln_test.sh, then
    
    sh scripts/gln_test.sh

Once trained, a sample usage of the GLN proposer API is 

    python sample_gln_proposer.py
Refer to sample_gln_proposer.py and modify accordingly for your own use case.

## Transformer
Based on:  
OpenNMT (https://opennmt.net/OpenNMT-py/)  
Molecular Transformer (https://github.com/pschwllr/MolecularTransformer)  
Bigchem/Karpov (https://github.com/bigchem/retrosynthesis)

Step 1. Prepare the raw SMILES .txt/.smi files for train, validation and test.
See https://github.com/bigchem/retrosynthesis/tree/master/data
for sample data format.

Step 2. Preprocessing: modify the args in scripts/transformer_preprocess.sh, then

    sh scripts/transformer_preprocess.sh

Step 3. Training: modify the args in scripts/transformer_train.sh, then
    
    sh scripts/transformer_train.sh

Step 4 (optional). Testing: modify the args in scripts/transformer_test.sh, then
    
    sh scripts/transformer_test.sh
    
NOTE: DO NOT change flags marked as "do_not_change_this"

Once trained, a sample usage of the Transformer proposer API is 

    python sample_transformer_proposer.py
Refer to sample_transformer_proposer.py and modify accordingly for your own use case.