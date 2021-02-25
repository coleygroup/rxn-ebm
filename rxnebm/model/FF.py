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
        super().__init__(hidden_sizes, dropout, activation, output_size, rctfp_size, prodfp_size, rxn_type)
        self.model_repr = "FeedforwardSingle"

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

    # ztu on 201211: commented out and modified expt.py accordingly
    # as a general advice do not override __repr__. Pytorch module relies on __repr__ to get model summary
    # def __repr__(self):
    #     return "FeedforwardEBM"  # needed by experiment.py for saving model details

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

    hidden_sizes : List[int]
        list of hidden layer sizes for the encoder, from layer 0 onwards 
        e.g. [1024, 512, 256] = layer 0 has 1024 neurons, layer 1 has 512 neurons etc.
    output_size : Optional[int] (Default = 1)
        how many outputs the model should give. for binary classification, this is just 1

    TODO: bayesian optimisation
    """

    def __init__(self, args):
        super().__init__()
        self.model_repr = "FeedforwardTriple3indiv3prod1cos"

        if args.rxn_type == "hybrid_all": # [rcts_fp, prod_fp, diff_fp]
            self.rctfp_size = args.rctfp_size
            self.prodfp_size = args.prodfp_size
            self.difffp_size = args.difffp_size
        else:
            raise ValueError(f'Not compatible with {args.rxn_type} fingerprints! Only works with hybrid_all')

        self.encoder_rcts = self.build_encoder(
            args.encoder_dropout, args.encoder_activation, args.encoder_hidden_size, self.rctfp_size
        )
        self.encoder_prod = self.build_encoder(
            args.encoder_dropout, args.encoder_activation, args.encoder_hidden_size, self.prodfp_size
        )
        self.encoder_diff = self.build_encoder(
            args.encoder_dropout, args.encoder_activation, args.encoder_hidden_size, self.difffp_size
        )

        if len(args.out_hidden_sizes) > 0:
            self.output_layer = self.build_encoder(
                args.out_dropout, args.out_activation, args.out_hidden_sizes, args.encoder_hidden_size[-1] * 6 + 1,
                output=True
            )
        else:
            self.output_layer = nn.Sequential(
                                    *[
                                    nn.Dropout(args.out_dropout), 
                                    nn.Linear(args.encoder_hidden_size[-1] * 6 + 1, 1) 
                                    ]
                                )

        model_utils.initialize_weights(self)

    def build(self):
      pass 

    def build_encoder(
        self,
        dropout: float,
        activation: str,
        hidden_sizes_encoder: List[int],
        input_dim: int,
        output: bool = False
    ):
        num_layers = len(hidden_sizes_encoder)
        activation = model_utils.get_activation_function(activation)
        ffn = [
                nn.Linear(input_dim, hidden_sizes_encoder[0])
            ]
        for i, layer in enumerate(range(num_layers - 1)):
            ffn.extend(
                [
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_sizes_encoder[i], hidden_sizes_encoder[i + 1]),
                ]
                )
        if output:
            ffn.extend(
                [
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_sizes_encoder[-1], 1),
                ]
                )
        return nn.Sequential(*ffn)

    def forward(self, batch: Tensor, probs: Optional[Tensor]=None) -> Tensor:
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


class FeedforwardMixture(nn.Module): 
    """
    Takes a product fingerprint as input and predicts whether each of the
    N retrosynthesis models ('Mixture of Experts') could successfully predict
    the ground truth precursors within some top-K predictions (e.g. K = 200)
    
    Examples of such experts include: GLN, Retrosim & RetroXpert

    hidden_sizes : List[int]
        list of hidden layer sizes for the encoder, from layer 0 onwards 
        e.g. [1024, 512, 256] = layer 0 has 1024 neurons, layer 1 has 512 neurons etc.
    """

    def __init__(self, args):
        super().__init__()
        self.model_repr = "FeedforwardMixture"
        self.encoder_prod = self.build_layers(
            args.encoder_dropout, args.encoder_activation, args.encoder_hidden_size, args.prodfp_size
        )

        self.output_layer = self.build_layers(
            args.out_dropout, args.out_activation, args.out_hidden_sizes, args.encoder_hidden_size[-1],
            output=True
        )
        model_utils.initialize_weights(self)

    def build(self):
      pass 

    def build_layers(
        self,
        dropout: float,
        activation: str,
        hidden_sizes: List[int],
        input_dim: int,
        output: bool = False
    ):
        num_layers = len(hidden_sizes)
        activation = model_utils.get_activation_function(activation)
        ffn = [
                nn.Linear(input_dim, hidden_sizes[0])
            ]
        for i, layer in enumerate(range(num_layers - 1)):
            ffn.extend(
                [
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                ]
                )
        if output:
            ffn.extend(
                [
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_sizes[-1], 3), # output dim = 3
                ]
                )
        return nn.Sequential(*ffn)

    def forward(self, batch: Tensor) -> Tensor:
        prod_embedding = self.encoder_prod(batch)                                   # N x embed_dim (hidden_sizes_encoder[-1])
        return self.output_layer(prod_embedding).squeeze(dim=1)                     # N x 1 x 3 => N x 3