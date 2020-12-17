#!/bin/bash

python trainEBM.py \
  --model_name="TransformerEBM" \
  --rxn_smis_file_prefix="50k_clean_rxnsmi_noreagent" \
  --onthefly \
  --do_pretrain \
  --do_test \
  --do_get_energies_and_acc \
	--log_file=s2e_rdm_2_mut_13.log \
	--expt_name=s2e_rdm_2_mut_13 \
	--vocab_file="vocab.txt" \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--max_seq_len=256 \
	--batch_size=64 \
	--minibatch_size=16 \
	--learning_rate=5e-3 \
	--optimizer="Adam" \
	--epochs=1 \
	--early_stop \
	--early_stop_min_delta=1e-4 \
	--early_stop_patience=2 \
	--num_workers=0 \
	--checkpoint
