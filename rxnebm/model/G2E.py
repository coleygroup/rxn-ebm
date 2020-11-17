import torch
import torch.nn as nn
from typing import Tuple
from rxnebm.data.chem_utils import ATOM_FDIM, BOND_FDIM
from rxnebm.model import model_utils


def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf


def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


class GRU(nn.Module):
    """GRU Message Passing layer."""

    def __init__(self,
                 input_size: int,
                 h_size: int,
                 depth: int,
                 device: str = 'cpu'):
        """
        Parameters
        ----------
        input_size: int,
            Size of the input
        h_size: int,
            Hidden state size
        depth: int,
            Number of time steps of message passing
        device: str, default cpu
            Device used for training
        """
        super().__init__()
        self.h_size = h_size
        self.input_size = input_size
        self.depth = depth
        self.device = device
        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.h_size, self.h_size).to(self.device)
        self.W_r = nn.Linear(self.input_size, self.h_size, bias=False).to(self.device)
        self.U_r = nn.Linear(self.h_size, self.h_size).to(self.device)
        self.W_h = nn.Linear(self.input_size + self.h_size, self.h_size).to(self.device)

    def get_init_state(self, fmess: torch.Tensor, init_state: torch.Tensor = None) -> torch.Tensor:
        """Get the initial hidden state of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        init_state: torch.Tensor, default None
            Custom initial state supplied.
        """
        h = torch.zeros(len(fmess), self.h_size, device=fmess.device)
        return h if init_state is None else torch.cat((h, init_state), dim=0)

    def get_hidden_state(self, h: torch.Tensor) -> torch.Tensor:
        """Gets the hidden state.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state of the GRU
        """
        return h

    def GRU(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """Implements the GRU gating equations.

        Parameters
        ----------
        x: torch.Tensor,
            Input tensor
        h_nei: torch.Tensor,
            Hidden states of the neighbors
        """
        sum_h = h_nei.sum(dim=1)
        z_input = torch.cat([x, sum_h], dim=1)
        z = torch.sigmoid(self.W_z(z_input))

        r_1 = self.W_r(x).view(-1, 1, self.h_size)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)

        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)
        h_input = torch.cat([x, sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * sum_h + z * pre_h
        return new_h

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size()[0], self.h_size, device=fmess.device)
        mask = torch.ones(h.size()[0], 1, device=h.device)
        mask[0, 0] = 0      # first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
        return h

    def sparse_forward(self, h: torch.Tensor, fmess: torch.Tensor,
                       submess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Unknown use.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state tensor
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        submess: torch.Tensor,
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        mask = h.new_ones(h.size()[0]).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            sub_h = self.GRU(fmess, h_nei)
            h = index_scatter(sub_h, h, submess)
        return h


class MPNEncoder(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings."""

    def __init__(self,
                 rnn_type: str,
                 input_size: int,
                 node_fdim: int,
                 h_size: int,
                 depth: int):
        """
        Parameters
        ----------
        rnn_type: str,
            Type of RNN used (gru/lstm)
        input_size: int,
            Input size
        node_fdim: int,
            Number of node features
        h_size: int,
            Hidden state size
        depth: int,
            Number of time steps in the RNN
        """
        super().__init__()
        self.h_size = h_size
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.depth = depth
        self.node_fdim = node_fdim
        self._build_layers()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNEncoder."""
        self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.h_size, self.h_size), nn.ReLU())
        if self.rnn_type == 'gru':
            self.rnn = GRU(self.input_size, self.h_size, self.depth)
        elif self.rnn_type == 'lstm':
            self.rnn = LSTM(self.input_size, self.h_size, self.depth)
        else:
            raise ValueError('unsupported rnn cell type ' + self.rnn_type)

    def forward(self, fnode: torch.Tensor, fmess: torch.Tensor,
                agraph: torch.Tensor, bgraph: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the MPNEncoder.

        Parameters
        ----------
        fnode: torch.Tensor,
            Node feature tensor
        fmess: torch.Tensor,
            Message features
        agraph: torch.Tensor,
            Neighborhood of an atom
        bgraph: torch.Tensor,
            Neighborhood of a bond, except the directed bond from the destination
            node to the source node
        mask: torch.Tensor,
            Masks on nodes
        """
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0      # first node is padding

        return node_hiddens * mask, h


class GraphFeatEncoder(nn.Module):
    """
    GraphFeatEncoder encodes molecules by using features of atoms and bonds,
    instead of a vocabulary, which is used for generation tasks.
    """

    def __init__(self,
                 n_atom_feat: int,
                 n_bond_feat: int,
                 rnn_type: str,
                 h_size: int,
                 depth: int):
        """
        Parameters
        ----------
        n_atom_feat: int,
            Number of atom features
        n_bond_feat: int,
            Number of bond features
        rnn_type: str,
            Type of RNN used for encoding
        h_size: int,
            Hidden state size
        depth: int,
            Number of time steps in the RNN
        """
        super().__init__()
        self.n_atom_feat = n_atom_feat
        self.n_bond_feat = n_bond_feat
        self.rnn_type = rnn_type
        self.atom_size = n_atom_feat
        self.h_size = h_size
        self.depth = depth

        self._build_layers()

    def _build_layers(self):
        """Build layers associated with the GraphFeatEncoder."""
        self.encoder = MPNEncoder(rnn_type=self.rnn_type,
                                  input_size=self.n_atom_feat + self.n_bond_feat,
                                  node_fdim=self.atom_size,
                                  h_size=self.h_size, depth=self.depth)

    @staticmethod
    def embed_graph(graph_tensors: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Replaces input graph tensors with corresponding feature vectors.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details,
            atom graph and bond graph for encoding neighborhood connectivity.
        """
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        hnode = fnode.clone()
        fmess1 = hnode.index_select(index=fmess[:, 0].long(), dim=0)
        fmess2 = fmess[:, 2:].clone()
        hmess = torch.cat([fmess1, fmess2], dim=-1)
        return hnode, hmess, agraph, bgraph

    def forward(self, graph_tensors: Tuple[torch.Tensor], scopes) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the graph encoder. First the feature vectors are extracted,
        and then encoded.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details,
            atom graph and bond graph for encoding neighborhood connectivity.
        scopes: Tuple[List]
            Scopes is composed of atom and bond scopes, which keep track of
            atom and bond indices for each molecule in the 2D feature list
        """
        tensors = self.embed_graph(graph_tensors)
        hatom, _ = self.encoder(*tensors, mask=None)
        atom_scope, bond_scope = scopes

        if isinstance(atom_scope[0], list):
            hmol = [torch.stack([hatom[st:st+le].sum(dim=0) for (st, le) in scope])
                    for scope in atom_scope]
        else:
            hmol = torch.stack([hatom[st:st+le].sum(dim=0) for st, le in atom_scope])
        return hatom, hmol


class G2E(nn.Module):
    def __init__(self, encoder_hidden_size, encoder_depth):
        super().__init__()
        self.encoder = GraphFeatEncoder(n_atom_feat=ATOM_FDIM,
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type="gru",
                                        h_size=encoder_hidden_size,
                                        depth=encoder_depth)
        model_utils.initialize_weights(self)

    def __repr__(self):
        return "GraphEBM"  # needed by experiment.py for saving model details

    def forward(self, batch):
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        graph_tensors, scopes = batch
        hatom, _ = self.encoder(graph_tensors=graph_tensors,
                                scopes=scopes)                  # [n_atoms, 400]
        atom_scope = scopes[0]                                  # [b*K, n_components, 2]
        molecular_lengths = [scope[-1][0] + scope[-1][1] - scope[0][0]
                          for scope in atom_scope]              # [b]

        energies = self.ffn(batch)  # tensor of size N x K x 1
        return energies.squeeze(dim=-1)  # scores: N x K after squeezing
