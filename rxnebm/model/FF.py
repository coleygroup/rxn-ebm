from typing import List, Optional

import torch
import torch.nn as nn

from rxnebm.model import base, model_utils

Tensor = torch.Tensor

class FeedforwardSingle(base.Feedforward): 
    """
    Currently supports 2 feedforward variants:
        diff: takes as input a difference FP of fp_size & fp_radius
        sep: takes as input a concatenation of [reactants FP, product FP]

    hidden_sizes : List[int]
        list of hidden layer sizes from layer 0 onwards
        e.g. [1024, 512, 256] = layer 0 has 1024 neurons, layer 1 has 512 neurons etc.
    output_size : Optional[int] (Default = 1)
        how many outputs the model should give. 
        for EBM, this must be 1, with no activation (to output scalar energy)

    TODO: bayesian optimisation
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        dropout: int,
        activation: str,
        output_size: Optional[int] = 1,
        rctfp_size: Optional[int] = 4096,
        prodfp_size: Optional[int] = 4096,
        rxn_type: Optional[str] = "diff",
        **kwargs
    ):
        super(FeedforwardSingle, self).__init__(hidden_sizes, dropout, activation, output_size, rctfp_size, prodfp_size, rxn_type)
        if rxn_type == "sep":
            input_dim = rctfp_size + prodfp_size
        elif rxn_type == "diff":
            input_dim = rctfp_size
            if rctfp_size != prodfp_size:
                raise ValueError(
                    "rctfp_size must equal prodfp_size for difference fingerprints!"
                )

        num_layers = len(hidden_sizes) + 1
        dropout = nn.Dropout(dropout)
        activation = model_utils.get_activation_function(activation)
        self.build(
            dropout, activation, hidden_sizes, input_dim, output_size, num_layers
        )
        model_utils.initialize_weights(self)

    def __repr__(self):
        return "FeedforwardEBM"  # needed by experiment.py for saving model details

    def build(
        self,
        dropout: nn.Dropout,
        activation: nn.Module,
        hidden_sizes: List[int],
        input_dim: int,
        output_size: int,
        num_layers: int,
    ):
        if num_layers == 1:
            ffn = [nn.Linear(input_dim, output_size)]
        else:
            ffn = [nn.Linear(input_dim, hidden_sizes[0])]

            # intermediate hidden layers
            for i, layer in enumerate(range(num_layers - 2)):
                ffn.extend(
                    [
                        activation,
                        dropout,
                        nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    ]
                )

            # last hidden layer
            ffn.extend(
                [
                    activation,
                    dropout,
                    nn.Linear(hidden_sizes[-1], output_size),
                ]
            )
        self.ffn = nn.Sequential(*ffn)

    def forward(self, batch: Tensor) -> Tensor:
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        energies = self.ffn(batch)  # tensor of size N x K x 1
        return energies.squeeze(dim=-1)  # N x K after squeezing


class FeedforwardTriple3indiv3prod1cos(nn.Module): 
    """
    Only supports
      sep: takes as input a tuple (reactants_fp, product_fp)

    hidden_sizes_encoder : List[int]
        list of hidden layer sizes for the encoder, from layer 0 onwards 
        e.g. [1024, 512, 256] = layer 0 has 1024 neurons, layer 1 has 512 neurons etc.
    output_size : Optional[int] (Default = 1)
        how many outputs the model should give. for binary classification, this is just 1

    TODO: bayesian optimisation
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        dropout: int,
        activation: str,
        output_size: Optional[int] = 1,
        rctfp_size: Optional[int] = 16384,
        prodfp_size: Optional[int] = 16384,
        difffp_size: Optional[int] = 16384,
        rxn_type: Optional[str] = "hybrid_all",
        **kwargs
    ):
      super(FeedforwardFingerprintTriple3indiv3prod1cos, self).__init__()
      if rxn_type == "hybrid_all": # [rcts_fp, prod_fp, diff_fp]
        self.rctfp_size = rctfp_size
        self.prodfp_size = prodfp_size
        self.difffp_size = difffp_size
      else:
        raise ValueError(f'Not compatible with {rxn_type} fingerprints! Only works with hybrid_all')

      num_layers = len(hidden_sizes)
      dropout_layer = nn.Dropout(dropout)
      activation = model_utils.get_activation_function(activation)
      self.encoder_rcts = self.build_encoder(
          dropout_layer, activation, hidden_sizes, rctfp_size, num_layers
      )
      self.encoder_prod = self.build_encoder(
          dropout_layer, activation, hidden_sizes, prodfp_size, num_layers
      )
      self.encoder_diff = self.build_encoder(
          dropout_layer, activation, hidden_sizes, difffp_size, num_layers
      )
      self.output_layer = nn.Sequential(*[
                                          dropout_layer, 
                                          nn.Linear(hidden_sizes[-1] * 6 + 1, output_size) 
                                        ])

      model_utils.initialize_weights(self)

    def __repr__(self):
        return "FeedforwardTriple3indiv3prod1cos"  # needed by experiment.py for saving model details

    def build(self):
      pass 

    def build_encoder(
        self,
        dropout: nn.Dropout,
        activation: nn.Module,
        hidden_sizes_encoder: List[int],
        input_dim: int, 
        num_layers: int,
    ):
        ffn = [nn.Linear(input_dim, hidden_sizes_encoder[0])]
        for i, layer in enumerate(range(num_layers - 1)):
            ffn.extend(
                [
                    activation,
                    dropout,
                    nn.Linear(hidden_sizes_encoder[i], hidden_sizes_encoder[i + 1]),
                ]
                )
        return nn.Sequential(*ffn)

    def forward(self, batch: Tensor) -> Tensor:
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        rcts_embedding = self.encoder_rcts(batch[:, :, :self.rctfp_size])                                   # N x K x embedding_dim (hidden_sizes_encoder[-1])
        prod_embedding = self.encoder_prod(batch[:, :, self.rctfp_size:self.rctfp_size+self.prodfp_size])   # N x K x embedding_dim 
        diff_embedding = self.encoder_diff(batch[:, :, self.rctfp_size+self.prodfp_size:])                  # N x K x embedding_dim 

        similarity = nn.CosineSimilarity(dim=-1)(rcts_embedding, prod_embedding).unsqueeze(dim=-1)          # N x K x 1
        
        combined_embedding = torch.cat([rcts_embedding, prod_embedding, diff_embedding,
                                        prod_embedding * rcts_embedding, diff_embedding * rcts_embedding, 
                                        diff_embedding * prod_embedding, similarity], dim=-1) 

        return self.output_layer(combined_embedding).squeeze(dim=-1)                                        # N x K 