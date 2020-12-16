#!/bin/bash

python trainEBM.py \
  --load_checkpoint \
  --model_name="GraphEBM" \
  --proposals_csv_file_prefix="retrosim_200maxtest_200maxprec" \
  --onthefly \
  --do_compute_graph_feat \
  --do_finetune \
  --do_test \
  --do_get_energies_and_acc \
	--log_file=g2e_rdm_2_mut_13_FINETUNE.log \
	--expt_name=g2e_rdm_2_mut_13_FINETUNE \
	--precomp_file_prefix="" \
	--old_expt_name=g2e_rdm_2_mut_13 \
	--expt_name=g2e_rdm_2_mut_13_FINETUNE \
	--date_trained=15_12_2020 \
	--representation="smiles" \
	--random_seed=0 \
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
