#!/bin/bash

conda activate rxnebm_ztu

# new
python3 trainEBM.py \
  --load_checkpoint \
  --model_name="TransformerEBM" \
  --rxn_smis_file_prefix="50k_clean_rxnsmi_noreagent" \
  --onthefly \
  --do_finetune \
  --do_test \
  --do_get_energies_and_acc \
	--log_file=s2e_rdm_2_mut_13_128emb_4layers_128hidden_8heads_256filter_drop20_10_mean.log \
	--expt_name=s2e_rdm_2_mut_13_128emb_4layers_128hidden_8heads_256filter_drop20_10_mean \
    --old_expt_name=s2e_rdm_2_mut_13_128emb_4layers_128hidden_8heads_256filter_drop20_10_mean \
    --date_trained=30_12_2020 \
	--vocab_file="vocab.txt" \
    --proposals_csv_file_prefix="retrosim_200maxtest_200maxprec" \
	--precomp_file_prefix="" \
	--representation="smiles" \
	--random_seed=0 \
	--max_seq_len=256 \
	--batch_size=8 \
	--minibatch_size=50 \
    --topk=49 \
    --maxk=50 \
	--learning_rate=1e-3 \
	--optimizer="Adam" \
	--epochs=100 \
	--early_stop \
	--early_stop_patience=2 \
    --lr_scheduler="ReduceLROnPlateau" \
	--lr_scheduler_factor=0.2 \
	--lr_scheduler_patience=0 \
	--num_workers=0 \
	--checkpoint \
	--drop_last

# original, by zhengkai
# python trainEBM.py \
#   --load_checkpoint \
#   --model_name="TransformerEBM" \
#   --proposals_csv_file_prefix="retrosim_200maxtest_200maxprec" \
#   --onthefly \
#   --do_finetune \
#   --do_test \
#   --do_get_energies_and_acc \
# 	--log_file=s2e_rdm_2_mut_13_FINETUNE.log \
# 	--expt_name=s2e_rdm_2_mut_13_FINETUNE \
# 	--vocab_file="vocab.txt" \
# 	--precomp_file_prefix="" \
# 	--old_expt_name=s2e_rdm_2_mut_13 \
# 	--expt_name=s2e_rdm_2_mut_13_FINETUNE \
# 	--date_trained=16_12_2020 \
# 	--representation="smiles" \
# 	--random_seed=0 \
# 	--max_seq_len=256 \
# 	--batch_size=64 \
# 	--minibatch_size=16 \
# 	--learning_rate=5e-3 \
# 	--optimizer="Adam" \
# 	--epochs=1 \
# 	--early_stop \
# 	--early_stop_min_delta=1e-4 \
# 	--early_stop_patience=2 \
# 	--num_workers=0 \
# 	--checkpoint
