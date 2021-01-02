#!/bin/bash

conda activate rxnebm_ztu

python3 trainEBM.py \
  --model_name="TransformerEBM" \
  --rxn_smis_file_prefix="50k_clean_rxnsmi_noreagent" \
  --onthefly \
  --do_pretrain \
  --do_test \
  --do_get_energies_and_acc \
	--log_file=s2e_rdm_2_mut_13_128emb_4layers_128hidden_8heads_256filter_drop20_10_mean.log \
	--expt_name=s2e_rdm_2_mut_13_128emb_4layers_128hidden_8heads_256filter_drop20_10_mean \
	--vocab_file="vocab.txt" \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--max_seq_len=256 \
	--batch_size=72 \
	--minibatch_size=16 \
	--learning_rate=5e-3 \
	--optimizer="Adam" \
	--epochs=50 \
	--early_stop \
	--early_stop_patience=3 \
    --lr_scheduler="ReduceLROnPlateau" \
	--lr_scheduler_factor=0.2 \
	--lr_scheduler_patience=1 \
	--num_workers=0 \
	--checkpoint \
	--drop_last

read -p "Press any key to continue" x