# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm_g2e

# for multi-GPU, choose any number (0-65535) for port
ebm_seed=$1
gln_seed=$2
python trainEBM.py \
  --ddp \
  --nodes 1 \
  --gpus 4 \
  --nr 0 \
  --port 51999 \
  --model_name "GraphEBM_2MPN" \
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
  --out_hidden_sizes 600 300 \
  --out_activation "PReLU" \
  --out_dropout 0.15 \
  --cache_suffix '50top_200max_stereo' \
  --do_train \
  --do_test \
  --do_get_energies_and_acc \
    --log_file 10x300_320_256x200_600x300_GLN_${ebm_seed}_2MPN_50top200max_bs2_1e4_fac30_pat1_stop3_f1e7_80ep_4GPU.log \
	--expt_name 10x300_320_256x200_600x300_GLN_${ebm_seed}_2MPN_50top200max_bs2_1e4_fac30_pat1_stop3_f1e7_80ep_4GPU \
    --proposals_csv_file_prefix "GLN_200topk_200maxk_noGT_${gln_seed}" \
	--representation "graph" \
	--random_seed ${ebm_seed} \
	--batch_size 2 \
    --batch_size_eval 4 \
	--minibatch_size 50 \
	--minibatch_eval 200 \
    --grad_clip 20 \
    --lr_floor 1e-7 \
	--learning_rate 1e-4 \
    --lr_scheduler_factor 0.3 \
    --lr_scheduler_patience 1 \
    --early_stop \
    --early_stop_patience 3 \
    --epochs 80 \
	--checkpoint \
    --test_on_train