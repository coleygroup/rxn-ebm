#!/bin/sh 
cd ..
python finetuneEBM.py \
	--log_file=rdm_5_cos_5_bit_5_1_1_mut_10_FINETUNE.log \
	--model_name=FeedforwardEBM \
	--old_expt_name=rdm_5_cos_5_bit_5_1_1_mut_10 \
	--expt_name=rdm_5_cos_5_bit_5_1_1_mut_10_FINETUNE \
	--precomp_file_prefix=retrosim_rxn_fps \
	--date_trained=02_11_2020 \
	--batch_size=512 \
	--learning_rate=8e-4 \
	--optimizer="Adam" \
	--epochs=100 \
	--early_stop \
	--early_stop_min_delta=1e-4 \
	--early_stop_patience=4 \
	--lr_scheduler=ReduceLROnPlateau \
	--lr_scheduler_factor=0.15 \
	--lr_scheduler_patience=0 \
	--num_workers=0 \
	--checkpoint \
	--random_seed=0

read -p "Press any key to continue" x