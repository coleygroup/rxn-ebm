#!/bin/sh 
python trainEBM.py \
	--log_file=g2e_rdm_2_mut_13.log \
	--expt_name=g2e_rdm_2_mut_13 \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--batch_size=32 \
	--learning_rate=5e-4 \
	--optimizer="Adam" \
	--epochs=5 \
	--early_stop \
	--early_stop_min_delta=1e-4 \
	--early_stop_patience=2 \
	--lr_scheduler="ReduceLROnPlateau" \
	--lr_scheduler_factor=0.3 \
	--lr_scheduler_patience=1 \
	--num_workers=0 \
	--checkpoint \
	--random_seed=0
