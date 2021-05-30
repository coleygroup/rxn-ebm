# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh

# ensure you are on a machine with CUDA, on a compute cluser you may need to load the modules as shown below
# module load gcc/8.3.0
# module load cuda/10.1
conda activate openretro

# input gln_seed accordingly, and also, best_ckpt to your best validation GLN checkpoint
gln_seed=$1
best_ckpt=$2

python gen_proposals/gen_gln.py \
    --log_file "gen_gln_${gln_seed}" \
    --output_folder "./rxnebm/data/cleaned_data/gln_${gln_seed}" \
    --model_path "./rxnebm/proposer/gln_openretro/checkpoints/gln_schneider50k_${gln_seed}/model-${best_ckpt}.dump/" \
    --input_file_prefix "50k_clean_rxnsmi_noreagent_allmapped_canon" \
    --propose \
    --compile \
    --topk 200 \
    --maxk 200 \
    --beam_size 200 \
    --checkpoint_every 999999

cd rxnebm/data/cleaned_data/gln_${gln_seed}/
{ 
    mv GLN_200topk_200maxk_200beam_train.csv ../GLN_200topk_200maxk_200beam_${gln_seed}_train.csv 
} && {
    mv GLN_200topk_200maxk_200beam_valid.csv ../GLN_200topk_200maxk_200beam_${gln_seed}_valid.csv
} && {
    mv GLN_200topk_200maxk_200beam_test.csv ../GLN_200topk_200maxk_200beam_${gln_seed}_test.csv
} && {
    cd ../
    rm gln_${gln_seed}/ -r
} && {
    cd ../../../
    python3 gen_proposals/gen_union_or_clean_proposals.py \
        --train \
        --valid \
        --test \
        --log_file "compile_gln_${gln_seed}" \
        --proposed_smi_file_prefixes "GLN_200topk_200maxk_200beam_${gln_seed}" \
        --proposers "GLN" \
        --topks 200 \
        --maxks 200 \
        --gln_seed ${gln_seed}
}

