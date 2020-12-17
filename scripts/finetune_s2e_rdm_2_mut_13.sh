#!/bin/bash

python trainEBM.py \
  --load_checkpoint \
  --model_name="TransformerEBM" \
  --proposals_csv_file_prefix="retrosim_200maxtest_200maxprec" \
  --onthefly \
  --do_finetune \
  --do_test \
  --do_get_energies_and_acc \
	--log_file=s2e_rdm_2_mut_13_FINETUNE.log \
	--expt_name=s2e_rdm_2_mut_13_FINETUNE \
	--vocab_file="vocab.txt" \
	--precomp_file_prefix="" \
	--old_expt_name=s2e_rdm_2_mut_13 \
	--expt_name=s2e_rdm_2_mut_13_FINETUNE \
	--date_trained=16_12_2020 \
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
