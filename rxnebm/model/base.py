from abc import ABC
from typing import List, Optional

import torch
import torch.nn as nn

from rxnebm.model import model_utils

Tensor = torch.Tensor

class Feedforward(nn.Module):
    """Abstract base class for feedforward neural networks"""

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
        super(Feedforward, self).__init__()
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
        raise NotImplementedError  # needed by experiment.py for saving model details

    def build(
        self,
        **kwargs
    ):
        """"""
        raise NotImplementedError

    def forward(self, batch: Tensor) -> Tensor:
        """"""
        raise NotImplementedError
