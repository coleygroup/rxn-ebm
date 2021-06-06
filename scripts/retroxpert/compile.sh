# ensure conda is properly initialised, such as by using the line below
# source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate rxnebm

seed=$1

# cd rxnebm/data/cleaned_data/results_${seed}/
cd rxnebm/proposer/retroxpert/OpenNMT-py/experiments/results_${seed}/
mv predictions_on_USPTO50K-aug-untyped_train-prediction_top200.csv ../../../../../data/cleaned_data/retroxpert_raw_200topk_200maxk_200beam_${seed}_train.csv
mv predictions_on_USPTO50K-aug-untyped_valid-prediction_top200.csv ../../../../../data/cleaned_data/retroxpert_raw_200topk_200maxk_200beam_${seed}_valid.csv
mv predictions_on_USPTO50K-aug-untyped_test-prediction_top200.csv ../../../../../data/cleaned_data/retroxpert_raw_200topk_200maxk_200beam_${seed}_test.csv

cd ../../../../../../

python gen_proposals/gen_retroxpert.py \
        --topk=200 \
        --maxk=200 \
        --beam_size=200 \
        --input_csv_prefix "retroxpert_raw_200topk_200maxk_200beam_${seed}" \
        --output_csv_prefix "retroxpert_200topk_200maxk_200beam_${seed}"

# additional cleaning step to ensure that no groundtruth proposals are in training data (possibly some RDKit bug causes 0.1% to still have)
{
    python gen_proposals/gen_union_or_clean_proposals.py \
            --train \
            --valid \
            --test \
            --log_file "clean_retroxpert_${seed}" \
            --proposed_smi_file_prefixes "retroxpert_200topk_200maxk_200beam_${seed}" \
            --proposers retroxpert \
            --topks 200 \
            --maxks 200 \
            --seed ${seed}
} && {
        rm rxnebm/data/cleaned_data/retroxpert_raw*
        rm rxnebm/data/cleaned_data/retroxpert*200beam*
}
# remove above files (no longer needed) only if the compilation succeeded