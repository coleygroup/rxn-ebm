# python3 gen_gln.py \
#     --propose \
#     --train \
#     --valid \
#     --test \
#     --topk=50 \
#     --beam_size=50 \
#     --maxk=50 \
#     --location=LOCAL

# python3 gen_gln.py \
#         --merge_chunks \
#         --phase_to_merge=train \
#         --topk=200 \
#         --maxk=200 \
#         --beam_size=200 \
#         --location=LOCAL \

python3 gen_gln.py \
        --compile \
        --train \
        --valid \
        --test \
        --topk=200 \
        --maxk=200 \
        --beam_size=200 \
        --location=LOCAL

read -p "Press any key to continue" x