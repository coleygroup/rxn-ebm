#!/bin/bash

python trainEBM.py \
  --model_name="G2E" \
  --do_compute_graph_feat \
  --do_pretrain \
  --do_test \
  --do_get_energies_and_acc \
	--log_file=g2e_rdm_2_mut_13.log \
	--expt_name=g2e_rdm_2_mut_13 \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--batch_size=64 \
	--learning_rate=5e-3 \
	--optimizer="Adam" \
	--epochs=1 \
	--num_workers=0 \
	--checkpoint
