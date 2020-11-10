#!/bin/sh 
cd .. 
python trainEBM.py \
	--log_file=rdm_5_cos_5_bit_5_1_1_mut_10.log \
	--expt_name=rdm_5_cos_5_bit_5_1_1_mut_10 \
	--precomp_file_prefix=50k_rdm_5_cos_5_bit_5_1_1_mut_10 \
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