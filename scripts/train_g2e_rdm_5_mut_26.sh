#!/bin/sh 
python trainEBM.py \
	--log_file=g2e_rdm_5_mut_26.log \
	--expt_name=g2e_rdm_5_mut_26 \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--batch_size=2048 \
	--learning_rate=5e-3 \
	--optimizer="Adam" \
	--epochs=30 \
	--early_stop \
	--early_stop_min_delta=1e-4 \
	--early_stop_patience=2 \
	--lr_scheduler="ReduceLROnPlateau" \
	--lr_scheduler_factor=0.3 \
	--lr_scheduler_patience=1 \
	--num_workers=0 \
	--checkpoint \
	--random_seed=0
  
read -p "Press any key to continue" x