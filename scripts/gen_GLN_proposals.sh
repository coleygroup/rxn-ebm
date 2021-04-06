python3 gen_proposals/gen_gln.py \
        --compile \
        --train \
        --valid \
        --test \
        --topk=200 \
        --maxk=200 \
        --beam_size=200 \
        --location=LOCAL

read -p "Press any key to continue" x