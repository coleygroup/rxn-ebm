# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm

# for multi-GPU, choose any number (0-65535) for port
ebm_seed=$1
xpert_seed=$2
python trainEBM.py \
  --ddp \
  --nodes 1 \
  --gpus 4 \
  --nr 0 \
  --port 31099 \
  --model_name "GraphEBM_1MPN" \
  --encoder_rnn_type gru \
  --atom_pool_type attention \
  --mol_pool_type sum \
  --encoder_hidden_size 300 \
  --encoder_inner_hidden_size 320 \
  --encoder_depth 10 \
  --encoder_dropout 0.08 \
  --encoder_activation 'ReLU' \
  --proj_hidden_sizes 256 200 \
  --proj_activation "PReLU" \
  --proj_dropout 0.12 \
  --preembed_size 90 \
  --cache_suffix '50top_200max_stereo' \
  --do_train \
  --do_test \
  --do_get_energies_and_acc \
    --log_file 90_10x300_320_256x200_XPERT_${ebm_seed}_1MPN_50top200max_bs4_3e4_fac60_pat1_stop3_f1e6_80ep_4GPU.log \
	--expt_name 90_10x300_320_256x200_XPERT_${ebm_seed}_1MPN_50top200max_bs4_3e4_fac60_pat1_stop3_f1e6_80ep_4GPU \
    --proposals_csv_file_prefix "retroxpert_200topk_200maxk_noGT_${xpert_seed}" \
	--representation "graph" \
	--random_seed ${ebm_seed} \
	--batch_size 4 \
    --batch_size_eval 8 \
	--minibatch_size 50 \
	--minibatch_eval 200 \
    --grad_clip 20 \
    --lr_floor_stop_training \
    --lr_floor 1e-6 \
	--learning_rate 3e-4 \
    --lr_scheduler_factor 0.6 \
    --lr_scheduler_patience 1 \
    --early_stop \
    --early_stop_patience 3 \
    --epochs 80 \
	--checkpoint \
    --test_on_train