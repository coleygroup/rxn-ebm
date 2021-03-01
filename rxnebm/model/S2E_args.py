S2E_args = {
    "embed_size": 128,
    "num_layers": 3,
    "hidden_size": 128,
    "num_heads": 4,
    "filter_size": 256,
    "dropout": 0.2,
    "attention_dropout": 0.1,
    "pooling_method": "CLS"
}
# 4x128_4h_256f_128e_drop20 has 434k params, takes 6 min on 8 GPUs for 50topk, bs=8
# could try CLS pooling
# seems like max_seq_len must match hidden_size???? 

# original
# S2E_args = {
#     "embed_size": 64,
#     "num_layers": 4,
#     "hidden_size": 64,
#     "num_heads": 4,
#     "filter_size": 256,
#     "dropout": 0.2,
#     "attention_dropout": 0.1,
#     "pooling_method": "mean"
# }