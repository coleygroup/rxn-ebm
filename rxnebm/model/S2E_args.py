S2E_args = {
    "embed_size": 256,
    "num_layers": 2,
    "hidden_size": 256,
    "num_heads": 16,
    "filter_size": 256,
    "dropout": 0.3,
    "attention_dropout": 0.1,
    "pooling_method": "mean"
}
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