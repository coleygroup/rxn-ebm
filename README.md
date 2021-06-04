# rxnebm
Energy-based modeling of chemical reactions

## Environmental setup
### Using conda
    # ensure conda is already initialized
    bash setup.sh
    conda activate rxnebm

## Data preparation / experimental setup
To get the results for our paper, we train each of the 4 one-step models on 3 seeds. Specifically, we used the following:
- GLN: 19260817, 20210423, 77777777
- RetroXpert: 11111111, 20210423, 77777777
- NeuralSym: 0, 20210423, 77777777
- RetroSim: no random seed needed

Thus, we have 3 sets of CSV files (train + valid + test) per one-step model. We then train one EBM re-ranker with a specified random seed (``ebm_seed``) on one set of CSV file, for a total of 3 repeats per one-step model. e.g. Graph-EBM (seed 0) on NeuralSym seed 0, Graph-EBM (seed 20210423) on NeuralSym seed 20210423, and Graph-EBM (seed 77777777) on NeuralSym seed 77777777. For GLN seed 19260817 and RetroXpert seed 11111111, we use ``ebm_seed = 0``. For RetroSim, we use ebm_seed of 0, 20210423, 77777777. We provide all 12 proposal CSV files on both [figshare](https://doi.org/10.6084/m9.figshare.14706267) and Google Drive ([here](https://drive.google.com/drive/u/1/folders/12tz9FX86zfOxwab0LYPILtJr5btqMwhu)). <br>

The training proposal CSV files are quite large (~200 MB), so please ensure you do have enough storage space (4.4 GB total). **Note** we have not uploaded fingerprints and graph features as these files are much larger. The graph features (train + val + test) can take up as much as 30 GB, while for fingerprints it is ~1 GB. See below in each proposer section for how to generate them yourself. If there is enough demand for us to upload these (very big) files, we may consider doing so.
 
## Training
Before training, ensure you have 1) the 3 CSV files 2) the 3 precomputed reaction data files (be it fingerprints, rxn_smi, graphs etc.). Refer to below for how we generate the reaction data files for a proposer. Note that ```<ebm_seed>``` refers to the random seed to be used for training the EBM re-ranker, and ```<proposer_seed>``` refers to the random seed that was used to train the one-step model. <br>
**Note:** As RetroSim has no random seed, you do not need to provide ```<proposer_seed>```.

If you are reloading a trained checkpoint for whatever reason, you additionally need to provide ```--old_expt_name <name>```, ```--date_trained <DD_MM_YYYY>``` and ```--load_checkpoint```. <br><br>
For FF-EBM

    bash scripts/<proposer>/FeedforwardEBM.sh <ebm_seed> <proposer_seed>
For Graph-EBM

    bash scripts/<proposer>/GraphEBM.sh <ebm_seed> <proposer_seed>
For Transformer-EBM (note that this yields poor results and we only report results on RetroSim). To train this, you just need the 3 CSV files, e.g. ``` rxnebm/data/cleaned_data/retrosim_200topk_200maxk_noGT_<phase>.csv ```

    bash scripts/retrosim/TransformerEBM.sh <ebm_seed>

## Cleaner USPTO-50K dataset
The data was obtained from [the dropbox folder](https://www.dropbox.com/sh/6ideflxcakrak10/AADN-TNZnuGjvwZYiLk7zvwra/schneider50k?dl=0&subfolder_nav_tracking=1) provided by the authors of [GLN](https://github.com/Hanjun-Dai/GLN). 
We renamed these 3 csv files from ```raw_{phase}.csv``` to ```'schneider50k_train.csv'```, ```'schneider50k_test.csv'``` and ```'schneider50k_valid.csv'```, and saved them to ```rxnebm/data/original_data``` (already included in this repo) <br>

For the re-ranking task, we trained four different retrosynthesis models. We use a single, extra-clean USPTO_50k dataset, split roughly into 80/10/10. These are derived from the three ``` schneider50k_{phase}.csv ``` files, using the script ```rxnebm/data/preprocess/clean_smiles.py```, i.e. 
```
    python -m rxnebm.data.preprocess.clean_smiles
```
This data has been included in this repository under ```rxnebm/data/cleaned_data/``` as ```50k_clean_rxnsmi_noreagent_allmapped_cano_{phase}.pickle```.  <br> 
**Note that these 3 .pickle files are extremely important, as we will use them as inputs to generate proposals & ground-truth for each one-step model.**

Specifically, we perform these steps:
1. Keep all atom mapping
2. Remove reaction SMILES strings with product molecules that are too small and clearly incorrect. The criteria used was ```len(prod_smi) < 3```. 4 reaction SMILES strings were caught by this criteria, with products: 		
    - ```'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccc([I:19])cc21>>[IH:19]'```
    - ```'O=C(CO)N1CCC(C(=O)[OH:28])CC1>>[OH2:28]'```
    - ```'CC(=O)[Br:4]>>[BrH:4]'```
    - ```'O=C(Cn1c(-c2ccc(Cl)c(Cl)c2)nc2cccnc21)[OH:10]>>[OH2:10]'```
3. Remove all duplicate reaction SMILES strings
4. Remove reaction SMILES in the training data that overlap with validation/test sets + validation data that overlap with the test set.
    - test_appears_in_train: 50
    - test_appears_in_valid: 6
    - valid_appears_in_train: 44
5. Finally, we obtain an (extra) clean dataset of reaction SMILES:
    - Train: 39713
    - Valid: 4989
    - Test: 5005
6. Canonicalization: After running ```clean_smiles.py```, we run ```canonicalize.py``` in the same folder:
    ```
        python -m rxnebm.data.preprocess.canonicalize
    ```
    <!-- For atom-mapped rxn_smi, there were 35/4/4 rxn_smi in train/valid/test changed due to RDKit canonicalization. -->
    
## Training and generating proposals for each one-step model
### Retrosim, with top-200 predictions (using 200 maximum precedents for product similarity search): 
- 3 CSV files of proposals from RetroSim.
    This is first generated by running 
    
    ``` python -m rxnebm.proposer.retrosim_model ```
    
    This step takes 13 hours on an 8 core machine. You only need to run this python script again if you wish to get more than top-200 predictions, or beyond 200 max precedents, or modify the underlying RetroSim model; otherwise, you can download it using ``` download_data.py ```.

    As a precaution, we canonicalize all these precursors again and ensure no training reaction has duplicate ground-truth, by running:

    ``` bash scripts/retrosim/clean.sh ```

- 3 .npz files of sparse reaction fingerprints ``` retrosim_rxn_fps_{phase}.npz ```

    This is generated by running
    
    ``` bash scripts/retrosim/make_fp.sh ```

    It takes about 8 minutes on a 32 core machine. Please refer to ```gen_proposals/gen_fps_from_proposals.py``` for detailed arguments.
    
    Since RetroSim will not generate the full 50/200 proposals for every product, we pad the reaction fingerprints with all-zero vectors for batching and mask these during training & testing.

- 3 sets of graph features (1 for each phase). Each set consists of: ```cache_feat_index.npz```,        ```cache_feat.npz```, ```cache_mask.pkl```, ```cache_smi.pkl```. Note that these 3 sets in total take up between 20 to 30 GBs, so ensure you have sufficient disk space. We again provide them in our Drive, but you can also generate them yourself using:

    ``` bash scripts/retrosim/make_graphfeat.sh ```

    It takes about 12 minutes on 32 cores.

- As stated in our paper, just training on the top-50 proposals (```--topk 50```) is sufficient and yields the same performance as training on more predictions (e.g. top-100/200); for testing, we still keep the top-200 proposals (```--maxk 200```) to maximize the chances of the published reaction appearing in those 200 proposals for re-ranking.

Once either the reaction fingerprints or the graphs have been generated, follow the instructions under ```Training``` above to train the EBMs.

### GLN, with top-200 predictions (beam_size=200)
- First we need to train GLN itself. We already include the 3 CSV files to train GLN, which contains the atom-mapped, canonicalized, extra-clean reaction SMILES from USPTO-50K, in 
``` rxnebm/proposer/gln_openretro/data/gln_schneider50k/ ```.
    - To generate these yourself, just run: <br>
    ``` python prep_data_for_retro_models.py --output_format gln```
    This takes as input the 3 .pickle files generated using ```clean_smiles.py``` above.

- We created a wrapper of the original GLN repo, to ease some issues installing GLN as a package, as well as standardize training, testing & proposing. Our official wrapper, **openretro**, is still under development to be released soon, and we include the GLN portion in this repo at: <br>
``` cd rxnebm/proposer/gln_openretro ```
- To install GLN: once you're in ``` gln_openretro ```, run: <br> ``` bash scripts/setup.sh ``` <br> This creates a conda environment called ```gln_openretro```, which you need to activate to train/test/propose with GLN. Note that the GLN authors compiled custom CUDA ops to speed up model training/testing, so you need to install GLN on a GPU machine with CUDA properly set up.
- To preprocess training data: <br>
``` bash scripts/preprocess.sh ```
- To train (takes ~2 hours on 1 RTX2080Ti). Note that you need to specify a training seed. <br>
``` bash scripts/train.sh <gln_seed> ```, e.g. ``` bash scripts/train.sh 0 ```

- To test (takes ~4 hours on 1 RTX2080Ti, because it tests all 10 checkpoints). Testing is important because it tells you the best checkpoint (by validation top-1 accuracy) to use for proposing. For example, on seed```77777777```, this should be ```model-6.dump```. <br>
``` bash scripts/test.sh <seed> ```

- To propose, we need to go back up to root with: <br>
``` cd ../../../ ``` <br>
Then run (takes ~12 hours on 1 RTX2080Ti): <br>
``` bash scripts/gln/propose_and_compile.sh <gln_seed> <best_ckpt> ``` <br>
You need to provide ``` gln_seed ``` and ``` best_ckpt ``` arguments. For example, if your best checkpoint was ```model-6.dump``` trained on seed ```77777777```, then: <br>
``` bash scripts/gln/propose_and_compile.sh 77777777 6 ``` <br>
which will output 3 cleaned CSV files in ``` rxnebm/data/cleaned_data ``` of the format ``` GLN_200topk_200maxk_noGT_<gln_seed>_<phase>.csv ```
- The last step is to generate either the fingerprints or graphs. This step is very similar across all 4 proposers. 
    - Fingerprints: <br>
    ``` bash scripts/gln/make_fp.sh <gln_seed> ```
    - Graphs: <br>
    ``` bash scripts/gln/make_graphfeat.sh <gln_seed> ```
- Finally, we can train the EBMs to re-rank GLN! Whew! That took a while. Alternatively, if you just want to reproduce our results, you can just grab the fingerprints and/or graphs of the proposals off our Google Drive.

### RetroXpert, with top-200 predictions (beam_size=200)
- To train RetroXpert, we include the 3 USPTO-50K CSV files in ``` rxnebm/proposer/retroxpert/data/USPTO50K/canonicalized_csv ```.
    - To generate these yourself, run: <br>
    ``` python prep_data_for_retro_models.py --output_format retroxpert```
    This takes as input the 3 .pickle files generated using ```clean_smiles.py``` above.
- We cloned the original RetroXpert repo, and added python scripts in order to generate proposals across the entire dataset. We also slightly modified the workflow to include a random seed for training. The folder is at: <br>
``` cd rxnebm/proposer/retroxpert ``` <br>
- To setup the environment for RetroXpert: once you're in ``` retroxpert ```, run: <br> ``` bash scripts/setup.sh ``` <br> This creates a conda environment called ```retroxpert```, which you need to activate to train/test/propose with RetroXpert.

- To preprocess training data. However, there is a slight RDKit bug in the template extraction step, where the same input data can generate different number of templates each time the script is run ``` python extract_semi_template_pattern.py --extract_pattern ```. <br> 
If you are simply reproducing our results, you do not need to extract the templates again, as we already include it in ``` data/USPTO50K/product_patterns.txt ```, which has 527 templates. So, you just need to run: <br>
``` bash scripts/preprocess.sh ```<br> and use 527 as ```<num_templates>``` for later steps. <br>
However, if you are using your own dataset, then you must extract the templates, by including the ```--extract_pattern``` flag when running ```preprocess.sh```: <br>
 ```bash scripts/preprocess.sh --extract_pattern```, and afterwards note down the EXACT number of templates extracted ``` <num_templates> ``` as you need to provide this to later scripts. <br>
- To train EGAT (~6 hours on 1 RTX2080Ti). You need to specify a training seed and number of extracted templates. <br>
``` bash scripts/train_EGAT.sh <retroxpert_seed> <num_templates> ```, e.g. ``` bash scripts/train_EGAT.sh 0 527 ```
- To train RGN (~30 hours on 2 RTX2080Ti). You need to specify a training seed. <br>
``` bash scripts/train_RGN.sh <retroxpert_seed> ```
- To translate using RGN and test the overall two-step performance (15 mins on 1 RTX2080Ti). We recommend running this step to double-check that the model is trained well without bugs. <br>
``` bash scripts/translate_RGN.sh <retroxpert_seed> ```
- The proposal stage contains two sub-steps. In the first sub-step, we directly use the existing RetroXpert code, only with minor modifications, to generate reactant-set predictions and evaluate the top-200 predictions. This sub-step takes ~8.5 hours on 1 RTX2080Ti: <br>
``` bash scripts/propose.sh <seed> <num_templates> ``` <br>
- In the second sub-step, because the output format from RetroXpert does not align with the input format we need for the EBM, we further process those top-200 proposals. This includes cleaning up invalid SMILES, de-duplicating proposals and ensuring there is only one ground-truth in each training reaction. This sub-step is much faster, ~10 mins on 8 cores (no GPU needed). <br>
First, go back to root: <br>
``` cd ../../../ ``` <br>
And then run: <br>
``` bash scripts/retroxpert/compile.sh <retroxpert_seed> ``` <br>
which will output 3 cleaned CSV files in ``` rxnebm/data/cleaned_data ``` of the format ``` retroxpert_200topk_200maxk_noGT_<retroxpert_seed>_<phase>.csv ```
- The last step is to generate either the fingerprints or graphs using those 3 cleaned CSV files. This step is very similar across all 4 proposers. 
    - Fingerprints: <br>
    ``` bash scripts/retroxpert/make_fp.sh <retroxpert_seed> ```
    - Graphs: <br>
    ``` bash scripts/retroxpert/make_graphfeat.sh <retroxpert_seed> ```
- Finally, we can train the EBMs to re-rank RetroXpert!

### NeuralSym, with top-200 predictions
- To train NeuralSym, we simply use the 3 .pickle files ``` 50k_clean_rxnsmi_noreagent_allmapped_canon_<phase>.pickle ``` generated using ```clean_smiles.py``` above, which contain the extra-cleaned USPTO-50K reactions. They've also been placed in NeuralSym's input data folder ``` rxnebm/proposer/neuralsym/data/ ```.
- As the original authors did not open-source NeuralSym, we re-implemented it from scratch following their paper and the repo is placed at: <br>
``` cd rxnebm/proposer/neuralsym ``` <br>
- To setup the environment for NeuralSym: once you're in ``` neuralsym ```, run: <br> ``` bash setup.sh ``` <br> This creates a conda environment called ```neuralsym```, which you need to activate to train/test/propose with NeuralSym.

- To preprocess training data into 32681-dim fingerprints. As we've heavily optimized this step, it takes only ~10 mins on 16 cores, and probably ~15-20 mins on 8 cores.  <br>
``` python prepare_data.py```. <br>

- To train (<5 mins on 1 RTX2080Ti, yep you read that correctly). Note that the accuracies used during training are template-matching accuracies, which are lower than reactant-matching accuracy (the actual metric for evaluating one-step retrosynthesis), because a particular reactant-set can be obtained from a given product through multiple templates. However, calculating template acucracy is faster (batchable) and more convenient, which is why we use it, given that our reactant-matching results are in agreement with literature values (in fact slightly better). <br>
You need to specify a training seed. <br>
``` bash scripts/train.sh <neuralsym_seed> ```, e.g. ``` bash scripts/train.sh 0 ```
- We combine the testing and proposing into a single step, and evaluate reactant-matching accuracy here. <br>
``` bash scripts/propose_and_compile.sh <neuralsym_seed> ``` <br>
This will output 3 cleaned CSV files in ``` rxnebm/data/cleaned_data ``` of the format ``` neuralsym_200topk_200maxk_noGT_<neuralsym_seed>_<phase>.csv ```
- Now, go back to root: <br>
``` cd ../../../ ``` <br>
The last step is to generate either the fingerprints or graphs using those 3 cleaned CSV files. This step is very similar across all 4 proposers. 
    - Fingerprints: <br>
    ``` bash scripts/neuralsym/make_fp.sh <neuralsym_seed> ```
    - Graphs: <br>
    ``` bash scripts/neuralsym/make_graphfeat.sh <neuralsym_seed> ```
- Finally, we can train the EBMs to re-rank NeuralSym!

### Union of GLN and RetroSim
- First, ensure you have generated the 3x proposal CSV files for both GLN and RetroSim by following the instructions for their respective sections above. This means you need both ``` GLN_200topk_200maxk_noGT_<gln_seed>_<phase>.csv ``` and ``` retrosim_200topk_200maxk_noGT_<phase>.csv ``` in ```rxnebm/data/cleaned_data```.

- To compile the union of proposals into a single CSV file for each phase, run: <br>
``` bash scripts/gln_sim/compile.sh <gln_seed> ``` <br>
which will output 3 cleaned CSV files in ``` rxnebm/data/cleaned_data ``` of the format ```GLN_50topk_200maxk_<gln_seed>_retrosim_50topk_200maxk_noGT_<phase>.csv ```.
- The last step is to generate either the fingerprints or graphs using those 3 cleaned CSV files. This step is very similar across all 4 proposers. 
    - Fingerprints: <br>
    ``` bash scripts/gln_sim/make_fp.sh <gln_seed> ```
    - Graphs: <br>
    ``` bash scripts/gln_sim/make_graphfeat.sh <gln_seed> ```
- Finally, we can train the Graph-EBM to re-rank the union of GLN and RetroSim! <br>
``` bash scripts/gln_sim/GraphEBM.sh <ebm_seed> <gln_seed> ```

### Citation
If you have used our code or referred to our paper, we would appreciate it if you could cite our work:
```
to be updated
```
