import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union
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
                 dropout: float = 0.1,
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
        self.dropout = dropout
        self.device = device
        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.h_size, self.h_size).to(self.device)
        self.W_r = nn.Linear(self.input_size, self.h_size, bias=False).to(self.device)
        self.U_r = nn.Linear(self.h_size, self.h_size).to(self.device)
        self.W_h = nn.Linear(self.input_size + self.h_size, self.h_size).to(self.device)
        self.Dropout = nn.Dropout(p=self.dropout)

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
            if i < self.depth - 1: # last iteration is w/o Dropout
                h = self.Dropout(h)
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
        for i in range(self.depth - 1):
            h_nei = index_select_ND(h, 0, bgraph)
            sub_h = self.GRU(fmess, h_nei)
            h = index_scatter(sub_h, h, submess)
        return h

class LSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 h_size: int,
                 depth: int,
                 dropout: float = 0.05,
                 **kwargs):
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
        super(LSTM, self).__init__(**kwargs)
        self.h_size = h_size
        self.input_size = input_size
        self.depth = depth
        self.dropout = dropout
        self._build_layer_components()

    def _build_layer_components(self):
        """Build layer components."""
        self.W_i = nn.Sequential(nn.Linear(self.input_size + self.h_size, self.h_size), nn.Sigmoid())
        self.W_o = nn.Sequential(nn.Linear(self.input_size + self.h_size, self.h_size), nn.Sigmoid())
        self.W_f = nn.Sequential(nn.Linear(self.input_size + self.h_size, self.h_size), nn.Sigmoid())
        self.W = nn.Sequential(nn.Linear(self.input_size + self.h_size, self.h_size), nn.Tanh())
        self.Dropout = nn.Dropout(p=self.dropout)

    def get_init_state(self, fmess: torch.Tensor,
                       init_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the initial hidden state of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        init_state: torch.Tensor, default None
            Custom initial state supplied.
        """
        h = torch.zeros(len(fmess), self.h_size, device=fmess.device)
        c = torch.zeros(len(fmess), self.h_size, device=fmess.device)
        if init_state is not None:
            h = torch.cat((h, init_state), dim=0)
            c = torch.cat((c, torch.zeros_like(init_state)), dim=0)
        return h, c

    def get_hidden_state(self, h: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Gets the hidden state.

        Parameters
        ----------
        h: Tuple[torch.Tensor, torch.Tensor],
            Hidden state tuple of the LSTM
        """
        return h[0]

    def LSTM(self, x: torch.Tensor, h_nei: torch.Tensor, c_nei: torch.Tensor) -> torch.Tensor:
        """Implements the LSTM gating equations.

        Parameters
        ----------
        x: torch.Tensor,
            Input tensor
        h_nei: torch.Tensor,
            Hidden states of the neighbors
        c_nei: torch.Tensor,
            Memory state of the neighbors
        """
        h_sum_nei = h_nei.sum(dim=1)
        x_expand = x.unsqueeze(1).expand(-1, h_nei.size(1), -1)
        i = self.W_i( torch.cat([x, h_sum_nei], dim=-1) )
        o = self.W_o( torch.cat([x, h_sum_nei], dim=-1) )
        f = self.W_f( torch.cat([x_expand, h_nei], dim=-1) )
        u = self.W( torch.cat([x, h_sum_nei], dim=-1) )
        c = i * u + (f * c_nei).sum(dim=1)
        h = o * torch.tanh(c)
        return h, c

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size(0), self.h_size, device=fmess.device)
        c = torch.zeros(fmess.size(0), self.h_size, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0                  # first message is padding

        for i in range(self.depth - 1):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            h, c = self.LSTM(fmess, h_nei, c_nei)
            if i < self.depth - 1: # last iteration is w/o Dropout
                h, c = self.Dropout(h), self.Dropout(c)
            h = h * mask
            c = c * mask
        
        return h, c

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
        h, c = h
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        c = c * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            sub_h, sub_c = self.LSTM(fmess, h_nei, c_nei)
            h = index_scatter(sub_h, h, submess)
            c = index_scatter(sub_c, c, submess)
        return h, c

class MPNEncoder(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings."""

    def __init__(self,
                 rnn_type: str,
                 input_size: int,
                 node_fdim: int,
                 h_size: int,
                 h_size_inner: Union[int, List[int]] = None, # minhtoo testing
                 preembed: bool = False, # minhtoo testing
                 preembed_size: Union[int, List[int]] = 80,
                 depth: int = 3,
                 dropout: float = 0.1,
                 encoder_activation = 'ReLU'
                 ):
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
        self.h_size_inner = h_size_inner
        self.encoder_activation = encoder_activation
        self.preembed = preembed
        self.preembed_size = preembed_size
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.depth = depth
        self.dropout = dropout
        self.node_fdim = node_fdim
        self._build_layers()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNEncoder."""
        encoder_activation = model_utils.get_activation_function(self.encoder_activation)
        if self.preembed:
            if isinstance(self.preembed_size, int):
                self.W_emb = nn.Sequential(
                        nn.Linear(self.input_size, self.preembed_size),
                    )
                rnn_input_size = self.preembed_size
            elif isinstance(self.preembed_size, list) and len(self.preembed_size) == 1:
                self.W_emb = nn.Sequential(
                        nn.Linear(self.input_size, self.preembed_size[0]),
                    )
                rnn_input_size = self.preembed_size[0]
            elif isinstance(self.preembed_size, list) and len(self.preembed_size) == 2:
                self.W_emb = nn.Sequential(
                        nn.Linear(self.input_size, self.preembed_size[0]),
                        encoder_activation,
                        nn.Dropout(self.dropout),
                        nn.Linear(self.preembed_size[0], self.preembed_size[1]),
                        encoder_activation,
                    )
                rnn_input_size = self.preembed_size[1]
            else:
                raise ValueError
            # then need to change input of RNN to self.preembed_size instead of self.input_size
        else:
            rnn_input_size = self.input_size

        if isinstance(self.h_size_inner, list) and len(self.h_size_inner) == 2: # 3 layers
            self.W_o = nn.Sequential(
                    nn.Linear(self.node_fdim + self.h_size, self.h_size_inner[0]), 
                    encoder_activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(self.h_size_inner[0], self.h_size_inner[1]), 
                    encoder_activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(self.h_size_inner[1], self.h_size),
                    encoder_activation
                )
        elif isinstance(self.h_size_inner, list) and len(self.h_size_inner) == 1: # 2 layers
            self.W_o = nn.Sequential(
                    nn.Linear(self.node_fdim + self.h_size, self.h_size_inner[0]), 
                    encoder_activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(self.h_size_inner[0], self.h_size),
                    encoder_activation
                )
        elif isinstance(self.h_size_inner, int): # 2 layers
            self.W_o = nn.Sequential(
                    nn.Linear(self.node_fdim + self.h_size, self.h_size_inner), 
                    encoder_activation,
                    # nn.Dropout(self.dropout), # remove dropout
                    nn.Linear(self.h_size_inner, self.h_size),
                    encoder_activation
                )
        else:
            self.W_o = nn.Sequential(
                    nn.Linear(self.node_fdim + self.h_size, self.h_size), 
                    encoder_activation
                )
        
        if self.rnn_type == 'gru':
            self.rnn = GRU(rnn_input_size, self.h_size, self.depth, self.dropout)
        elif self.rnn_type == 'lstm':
            self.rnn = LSTM(rnn_input_size, self.h_size, self.depth, self.dropout)
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
        if self.preembed:
            fmess = self.W_emb(fmess)
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
                 h_size_inner: Union[int, List[int]] = None,
                 preembed: bool = False,
                 preembed_size: Union[int, List[int]] = None,
                 depth: int = 3,
                 dropout: float = 0.1,
                 atom_pool_type: str = "sum",
                 encoder_activation: str = 'ReLU'
                 ):
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
        self.preembed = preembed
        self.preembed_size = preembed_size
        self.h_size = h_size
        self.h_size_inner = h_size_inner
        self.encoder_activation = encoder_activation
        self.depth = depth
        self.dropout = dropout
        self.atom_pool_type = atom_pool_type

        self._build_layers()

    def _build_layers(self):
        """Build layers associated with the GraphFeatEncoder."""
        self.encoder = MPNEncoder(rnn_type=self.rnn_type,
                                  input_size=self.n_atom_feat + self.n_bond_feat,
                                  node_fdim=self.atom_size,
                                  preembed=self.preembed,
                                  preembed_size=self.preembed_size,
                                  h_size=self.h_size,
                                  h_size_inner=self.h_size_inner,
                                  depth=self.depth,
                                  dropout=self.dropout,
                                  encoder_activation=self.encoder_activation)
        self.attn_hidden_1 = nn.Linear(self.h_size, self.h_size)
        self.elu = nn.ELU()
        self.attn_hidden_2 = nn.Linear(self.h_size, self.h_size)
        self.attn_output = nn.Linear(self.h_size * 2, self.h_size)

    def forward(self, graph_tensors: Tuple[torch.Tensor, ...], scopes) -> Tuple[torch.Tensor, ...]:
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
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        # logging.info('------------- agraph -------------')
        # logging.info(agraph)            # [905, 5]
        # logging.info(agraph.size())
        # logging.info(agraph[0])

        # logging.info('------------- bgraph -------------')
        # logging.info(bgraph)            # [1777, 4]
        # logging.info(bgraph.size())
        # logging.info(bgraph[0])

        # logging.info('------------- fmess -------------')
        # logging.info(fmess)             # [1777 ,8]
        # logging.info(fmess.size())
        # logging.info(fmess[0])

        atom_scope, bond_scope = scopes

        # embed graph, note that for directed graph, fmess[any, 0:2] = u, v
        hnode = fnode.clone()
        fmess1 = hnode.index_select(index=fmess[:, 0].long(), dim=0) # [1777, 99]
        # logging.info('------------- fmess1 -------------')
        # logging.info(fmess1)
        # logging.info(fmess1.size())
        # logging.info(fmess1[0])

        fmess2 = fmess[:, 2:].clone()       # [1777, 6]
        hmess = torch.cat([fmess1, fmess2], dim=-1)
        # logging.info('------------- hmess -------------')
        # logging.info(hmess)               # [1777, 105]
        # logging.info(hmess.size())
        # logging.info(hmess[0])

        # encode
        hatom, _ = self.encoder(hnode, hmess, agraph, bgraph, mask=None)

        hmol = []
        for scope in atom_scope:
            hmol_single = []
            for start, length in scope:
                h = hatom[start:start+length]

                if self.atom_pool_type == "sum":                # Sum pooling
                    hmol_single.append(h.sum(dim=0))
                elif self.atom_pool_type == "mean":             # Mean pooling
                    hmol_single.append(h.mean(dim=0))
                elif self.atom_pool_type == "attention":        # Attention pooling
                    h_mean = h.mean(dim=0)
                    attn_context = self.elu(self.attn_hidden_1(h))          # [length, h] -> [length, h]
                    attn_logit = self.attn_hidden_2(attn_context)           # [length, h] -> [length, h]

                    attn_weight = torch.exp(attn_logit)
                    attn_sum = torch.sum(h * attn_weight, dim=0)            # [length, h] -> [h]
                    attn_pool = attn_sum / attn_weight.sum(dim=0)

                    attn_pool = torch.cat([h_mean, attn_pool])              # [h] -> [h*2]
                    attn_pool = torch.tanh(self.attn_output(attn_pool))     # [h*2] -> [h]

                    hmol_single.append(attn_pool)
                else:
                    raise NotImplementedError(f"Unsupported atom_pool_type {self.atom_pool_type}! "
                                              f"Please use sum/mean/attention")

            hmol.append(torch.stack(hmol_single))

        """Deprecated
        # if isinstance(atom_scope[0], list):
        if True:
            for scope in atom_scope:
                # if not scope:
                if False:
                    hmol.append(torch.zeros([1, self.h_size], device=hatom.device))
                else:
                    hmol.append(torch.stack([hatom[st:st+le].sum(dim=0) for st, le in scope]))

            # hmol = [torch.stack([hatom[st:st+le].sum(dim=0) for (st, le) in scope])
            #         for scope in atom_scope]
        else:
            hmol = torch.stack([hatom[st:st+le].sum(dim=0) for st, le in atom_scope])
        """
        return hatom, hmol


class G2E(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_repr = "GraphEBM"
        self.args = args
        self.mol_pool_type = args.mol_pool_type

        if torch.cuda.is_available() and self.args.dataparallel:
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1
        
        if isinstance(args.encoder_hidden_size, list):
            assert len(args.encoder_hidden_size) == 1, 'MPN encoder_hidden_size must be a single integer!'
            args.encoder_hidden_size = args.encoder_hidden_size[0]
        self.encoder = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type=args.encoder_rnn_type,
                                        h_size=args.encoder_hidden_size,
                                        h_size_inner=args.encoder_inner_hidden_size,
                                        preembed=True if args.preembed_size is not None else False,
                                        preembed_size=args.preembed_size,
                                        depth=args.encoder_depth,
                                        dropout=args.encoder_dropout,
                                        encoder_activation=args.encoder_activation,
                                        atom_pool_type=args.atom_pool_type)

        out_activation = model_utils.get_activation_function(args.out_activation)
        self.output = self.build_output(
            out_activation, args.out_hidden_sizes, args.out_dropout,
            args.encoder_hidden_size * 4
        )

        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True
        logging.info("Initializing weights")
        model_utils.initialize_weights(self)

    def build_output(
        self,
        activation: nn.Module,
        hidden_sizes: List[int],
        dropout: float,
        input_dim: int,
    ):
        num_layers = len(hidden_sizes)
        ffn = [nn.Linear(input_dim, hidden_sizes[0])] # could add nn.Dropout(dropout) before this, if we wish
        for i, layer in enumerate(range(num_layers - 1)):
            ffn.extend(
                [
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                ]
                )
        ffn.extend(
            [
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[-1], 1), # output_dim = 1 to get energy value
            ]
        )
        return nn.Sequential(*ffn)

    def forward(self, batch, probs: Optional[torch.Tensor]=None):
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        graph_tensors, scopes, batch_size = batch

        # fnode, fmess, agraph, bgraph, _ = graph_tensors
        # logging.info("-------fnode-------")
        # logging.info(fnode)
        # logging.info(fnode[0])
        # logging.info(fnode[0].size())
        # logging.info(fnode.size())
        # logging.info("-------agraph-------")
        # logging.info(agraph)
        # logging.info(agraph[0])
        # logging.info(agraph[0].size())
        # logging.info(agraph.size())

        # atom_scope, bond_scope = scopes
        # logging.info("-------atom_scope-------")
        # logging.info(atom_scope)
        # logging.info(atom_scope[0])
        # logging.info("-------bond_scope-------")
        # logging.info(bond_scope)
        # logging.info(bond_scope[0])

        hatom, hmol = self.encoder(graph_tensors=graph_tensors,
                                   scopes=scopes)
        # logging.info("-------hatom-------")
        # logging.info(hatom[0])
        # logging.info(hatom.size())
        # logging.info(hatom[0].size())
        # logging.info("-------hmol-------")
        # logging.info(hmol[0])
        # logging.info(len(hmol))
        # logging.info([h.size() for h in hmol])

        # hatom: [n_atoms, 400], hmol: list of [n_molecules, 400]
        # n_molecules = batch_size * (r + p_pos + p_negs) = e.g. 2 * (1 + 1 + (5+26)) = 66
        # want energies to be 2 * 32
        # list of 66 [n_molecules, 400] => [2, 32]

        if self.mol_pool_type == "sum":
            hmol = [torch.sum(h, dim=0, keepdim=True) for h in hmol]        # list of [n_molecules, h] => list of [1, h]
        elif self.mol_pool_type == "mean":
            hmol = [torch.mean(h, dim=0, keepdim=True) for h in hmol]
        else:
            raise NotImplementedError(f"Unsupported mol_pool_type {self.mol_pool_type}! "
                                      f"Please use sum/mean")
        # logging.info([h.size() for h in hmol])

        batch_pooled_hmols = []
    
        mols_per_minibatch = len(hmol) // batch_size // self.num_devices  # = (1) r + (mini_bsz) p or (1) p + (mini_bsz) r
        # assert mols_per_minibatch == self.args.minibatch_size + 1, \
        #     f"calculated minibatch size: {mols_per_minibatch-1}, given in args: {self.args.minibatch_size}"

        # logging.info(f"{len(hmol)}, {batch_size}, {mols_per_minibatch}")
        for i in range(batch_size):
            if self.reactant_first:                         # (1) r + mini_bsz p
                r_hmol = hmol[i*mols_per_minibatch]                             # [1, h]
                r_hmols = r_hmol.repeat(mols_per_minibatch - 1, 1)              # [mini_bsz, h]
                p_hmols = hmol[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                p_hmols = torch.cat(p_hmols, 0)                                 # [mini_bsz, h]
            else:
                p_hmols = hmol[i*mols_per_minibatch]        # (1) p + mini_bsz (r)
                p_hmols = p_hmols.repeat(mols_per_minibatch - 1, 1)
                r_hmols = hmol[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                r_hmols = torch.cat(r_hmols, 0)

            diff = p_hmols - r_hmols # torch.abs(p_hmols - r_hmols)             # [mini_bsz, h]
            prod = r_hmols * p_hmols                                            # [mini_bsz, h]

            pooled_hmols = torch.cat([r_hmols, p_hmols, diff, prod], 1)         # [mini_bsz, h*4]
            pooled_hmols = torch.unsqueeze(pooled_hmols, 0)                     # [1, mini_bsz, h*4]

            batch_pooled_hmols.append(pooled_hmols)

        batch_pooled_hmols = torch.cat(batch_pooled_hmols, 0)                   # [bsz, mini_bsz, h*4]
        energies = self.output(batch_pooled_hmols)                              # [bsz, mini_bsz, 1]
        # logging.info("-------energies-------")
        # logging.info(energies)
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]


class G2ECross(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_repr = "GraphEBM_Cross"
        self.args = args
        self.mol_pool_type = args.mol_pool_type

        if torch.cuda.is_available() and self.args.dataparallel:
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type=args.encoder_rnn_type,
                                        h_size=args.encoder_hidden_size,
                                        h_size_inner=args.encoder_inner_hidden_size,
                                        preembed=True if args.preembed_size is not None else False,
                                        preembed_size=args.preembed_size,
                                        depth=args.encoder_depth,
                                        dropout=args.encoder_dropout,
                                        encoder_activation=args.encoder_activation,
                                        atom_pool_type=args.atom_pool_type)

        self.h_size = args.encoder_hidden_size

        self.SegmentEmbed = nn.Linear(6, self.h_size, bias=False)

        self.attn_hidden_1 = nn.Linear(self.h_size, self.h_size)
        self.elu = nn.ELU()
        self.attn_hidden_2 = nn.Linear(self.h_size, self.h_size)
        self.attn_output = nn.Linear(self.h_size * 2, self.h_size)

        self.output = nn.Linear(args.encoder_hidden_size, 1)
        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True
        logging.info("Initializing weights")
        model_utils.initialize_weights(self)

    def segment_embedding(self, side: str, idx: int):
        if side == "p":
            idx += 3
        one_hot_idx = torch.zeros(6, dtype=torch.float).cuda()
        one_hot_idx[idx] = 1

        return self.SegmentEmbed(one_hot_idx)

    def forward(self, batch, probs: Optional[torch.Tensor] = None):
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        graph_tensors, scopes, batch_size = batch
        hatom, _ = self.encoder(graph_tensors=graph_tensors,
                                scopes=scopes)
        atom_scope, bond_scope = scopes

        # Operates on hatom (vs. hmol) so it's pretty much blind to atom_pool_type and mol_pool_type

        mols_per_minibatch = len(atom_scope) // batch_size

        batch_pooled_hmols = []
        for i in range(batch_size):
            if self.reactant_first:
                raise NotImplementedError
            else:
                atom_scope_p = atom_scope[i*mols_per_minibatch]
                atom_scope_r = atom_scope[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]

                # product atom encodings with segment embedding
                h_p = []
                for segment_idx_p, (start, length) in enumerate(atom_scope_p):
                    segment_embedding = self.segment_embedding("p", segment_idx_p)
                    h_p.append(hatom[start:start+length] + segment_embedding)

                h_rxn = []
                for scope in atom_scope_r:
                    # reactant atom encodings with segment embedding, for each r in the minibatch
                    h = []
                    for segment_idx_r, (start, length) in enumerate(scope):
                        segment_embedding = self.segment_embedding("r", segment_idx_r)
                        h.append(hatom[start:start + length] + segment_embedding)
                    h.extend(h_p)               # combine r + p atoms to make it a 'cross-encoder'
                    h = torch.cat(h, dim=0)

                    # Attention pool over all atoms in the reaction
                    h_mean = h.mean(dim=0)
                    attn_context = self.elu(self.attn_hidden_1(h))          # [length, h] -> [length, h]
                    attn_logit = self.attn_hidden_2(attn_context)           # [length, h] -> [length, h]

                    attn_weight = torch.exp(attn_logit)
                    attn_sum = torch.sum(h * attn_weight, dim=0)            # [length, h] -> [h]
                    attn_pool = attn_sum / attn_weight.sum(dim=0)

                    attn_pool = torch.cat([h_mean, attn_pool])              # [h] -> [h*2]
                    attn_pool = torch.tanh(self.attn_output(attn_pool))     # [h*2] -> [h]

                    h_rxn.append(attn_pool)

                h_rxn = torch.stack(h_rxn, dim=0)                           # [mini_bsz, h]

            batch_pooled_hmols.append(h_rxn)

        batch_pooled_hmols = torch.stack(batch_pooled_hmols, 0)             # [bsz, mini_bsz, h]
        energies = self.output(batch_pooled_hmols)                          # [bsz, mini_bsz, 1]

        return energies.squeeze(dim=-1)

class G2E_projBoth(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_repr = "GraphEBM_projBoth"
        self.args = args
        self.mol_pool_type = args.mol_pool_type

        if torch.cuda.is_available() and self.args.dataparallel:
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type=args.encoder_rnn_type,
                                        h_size=args.encoder_hidden_size,
                                        h_size_inner=args.encoder_inner_hidden_size,
                                        preembed=True if args.preembed_size is not None else False,
                                        preembed_size=args.preembed_size,
                                        depth=args.encoder_depth,
                                        dropout=args.encoder_dropout,
                                        encoder_activation=args.encoder_activation,
                                        atom_pool_type=args.atom_pool_type)
        
        proj_activation = model_utils.get_activation_function(args.proj_activation)
        self.projection_r = self.build_projection(
            proj_activation, args.proj_hidden_sizes, args.proj_dropout, args.encoder_hidden_size # output_dim = proj_hidden_sizes[-1]
        )
        self.projection_p = self.build_projection(
            proj_activation, args.proj_hidden_sizes, args.proj_dropout, args.encoder_hidden_size # output_dim = proj_hidden_sizes[-1]
        )
        
        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True

    def build_projection(
        self,
        activation: nn.Module,
        hidden_sizes: List[int],
        dropout: float,
        input_dim: int,
    ):
        num_layers = len(hidden_sizes)
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
        return nn.Sequential(*ffn)

    def forward(self, batch, probs: Optional[torch.Tensor]=None):
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        graph_tensors, scopes, batch_size = batch

        # debug
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        if torch.isnan(torch.sum(fnode, (0, 1))) or torch.isnan(torch.sum(fmess, (0, 1))) or \
            torch.isnan(torch.sum(agraph, (0, 1))) or torch.isnan(torch.sum(bgraph, (0, 1))):
            raise ValueError('nan input found')
        # atom_scope, bond_scope = scopes # scopes is a list of np.arrays, hard to check efficiently

        hatom, hmol = self.encoder(graph_tensors=graph_tensors,
                                   scopes=scopes)

        if self.mol_pool_type == "sum":
            hmol = [torch.sum(h, dim=0, keepdim=True) for h in hmol]        # list of [n_molecules, h] => list of [1, h]
            for h in hmol:
                if torch.isnan(torch.sum(h, (0, 1))):
                    print('nan in hmol')
                    
        elif self.mol_pool_type == "mean":
            hmol = [torch.mean(h, dim=0, keepdim=True) for h in hmol]
        else:
            raise NotImplementedError(f"Unsupported mol_pool_type {self.mol_pool_type}! "
                                      f"Please use sum/mean")
        batch_pooled_r_mols = []
        batch_pooled_p_mols = []
    
        mols_per_minibatch = len(hmol) // batch_size // self.num_devices  # = (1) r + (mini_bsz) p or (1) p + (mini_bsz) r
        for i in range(batch_size):
            if self.reactant_first:                         # (1) r + mini_bsz p
                r_hmol = hmol[i*mols_per_minibatch]                             # [1, h]
                r_hmols = r_hmol.repeat(mols_per_minibatch - 1, 1)              # [mini_bsz, h]
                p_hmols = hmol[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                p_hmols = torch.cat(p_hmols, 0)                                 # [mini_bsz, h]
            else:
                p_hmols = hmol[i*mols_per_minibatch]        # (1) p + mini_bsz (r)
                p_hmols = p_hmols.repeat(mols_per_minibatch - 1, 1)
                r_hmols = hmol[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                r_hmols = torch.cat(r_hmols, 0)

            pooled_r_hmols = torch.unsqueeze(r_hmols, 0)                        # [1, mini_bsz, h]
            pooled_p_hmols = torch.unsqueeze(p_hmols, 0)                        # [1, mini_bsz, h]
            batch_pooled_r_mols.append(pooled_r_hmols)
            batch_pooled_p_mols.append(pooled_p_hmols)

        batch_pooled_r_mols = torch.cat(batch_pooled_r_mols, 0)                 # [bsz, mini_bsz, h]
        batch_pooled_p_mols = torch.cat(batch_pooled_p_mols, 0)                 # [bsz, mini_bsz, h]

        proj_pooled_r_mols = self.projection_r(
                                    batch_pooled_r_mols
                                ).unsqueeze(dim=-1)                             # [bsz, mini_bsz, h] => [bsz, mini_bsz, d] => [bsz, mini_bsz, d, 1]
        proj_pooled_p_mols = self.projection_p(
                                    batch_pooled_p_mols
                                ).unsqueeze(dim=-1)                             # [bsz, mini_bsz, h] => [bsz, mini_bsz, d] => [bsz, mini_bsz, d, 1]
        
        # dot product
        energies = torch.matmul(
                        torch.transpose(proj_pooled_p_mols, 2, 3),
                        proj_pooled_r_mols
                    ).squeeze(dim=-1)                                           # [bsz, mini_bsz, 1, d] x [bsz, mini_bsz, d, 1] => [bsz, mini_bsz, 1]
        
        # debug
        if torch.isnan(torch.sum(energies, (0, 1, 2))):
            print('nan in energies')
        
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]

class G2E_projBoth_FFout(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_repr = "GraphEBM_projBoth_FFout"
        self.args = args
        self.mol_pool_type = args.mol_pool_type

        if torch.cuda.is_available() and self.args.dataparallel:
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type=args.encoder_rnn_type,
                                        h_size=args.encoder_hidden_size,
                                        h_size_inner=args.encoder_inner_hidden_size,
                                        preembed=True if args.preembed_size is not None else False,
                                        preembed_size=args.preembed_size,
                                        depth=args.encoder_depth,
                                        dropout=args.encoder_dropout,
                                        encoder_activation=args.encoder_activation,
                                        atom_pool_type=args.atom_pool_type)
        
        proj_activation = model_utils.get_activation_function(args.proj_activation)
        out_activation = model_utils.get_activation_function(args.out_activation)
        self.projection_r = self.build_projection(
            proj_activation, args.proj_hidden_sizes, args.proj_dropout, args.encoder_hidden_size # output_dim = proj_hidden_sizes[-1]
        )
        self.projection_p = self.build_projection(
            proj_activation, args.proj_hidden_sizes, args.proj_dropout, args.encoder_hidden_size # output_dim = proj_hidden_sizes[-1]
        )
        self.output = self.build_output(
                out_activation, args.out_hidden_sizes, args.out_dropout,
                args.proj_hidden_sizes[-1] * 4
            )
        
        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True

    def build_output(
            self,
            activation: nn.Module,
            hidden_sizes: List[int],
            dropout: float,
            input_dim: int,
        ):
            num_layers = len(hidden_sizes)
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
            ffn.extend(
                [
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_sizes[-1], 1),
                ]
            )
            return nn.Sequential(*ffn)

    def build_projection(
        self,
        activation: nn.Module,
        hidden_sizes: List[int],
        dropout: float,
        input_dim: int,
    ):
        num_layers = len(hidden_sizes)
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
        return nn.Sequential(*ffn)

    def forward(self, batch, probs: Optional[torch.Tensor]=None):
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        graph_tensors, scopes, batch_size = batch

        # debug
        # fnode, fmess, agraph, bgraph, _ = graph_tensors
        # if torch.isnan(torch.sum(fnode, (0, 1))) or torch.isnan(torch.sum(fmess, (0, 1))) or \
        #     torch.isnan(torch.sum(agraph, (0, 1))) or torch.isnan(torch.sum(bgraph, (0, 1))):
        #     raise ValueError('nan input found')
        # atom_scope, bond_scope = scopes # scopes is a list of np.arrays, hard to check efficiently

        hatom, hmol = self.encoder(graph_tensors=graph_tensors,
                                   scopes=scopes)

        if self.mol_pool_type == "sum":
            hmol = [torch.sum(h, dim=0, keepdim=True) for h in hmol]        # list of [n_molecules, h] => list of [1, h]
            # for h in hmol:
            #     if torch.isnan(torch.sum(h, (0, 1))):
            #         print('nan in hmol')
                    
        elif self.mol_pool_type == "mean":
            hmol = [torch.mean(h, dim=0, keepdim=True) for h in hmol]
        else:
            raise NotImplementedError(f"Unsupported mol_pool_type {self.mol_pool_type}! "
                                      f"Please use sum/mean")
        batch_pooled_r_mols = []
        batch_pooled_p_mols = []
    
        mols_per_minibatch = len(hmol) // batch_size // self.num_devices  # = (1) r + (mini_bsz) p or (1) p + (mini_bsz) r
        for i in range(batch_size):
            if self.reactant_first:                         # (1) r + mini_bsz p
                r_hmol = hmol[i*mols_per_minibatch]                             # [1, h]
                r_hmols = r_hmol.repeat(mols_per_minibatch - 1, 1)              # [mini_bsz, h]
                p_hmols = hmol[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                p_hmols = torch.cat(p_hmols, 0)                                 # [mini_bsz, h]
            else:
                p_hmols = hmol[i*mols_per_minibatch]        # (1) p + mini_bsz (r)
                p_hmols = p_hmols.repeat(mols_per_minibatch - 1, 1)
                r_hmols = hmol[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                r_hmols = torch.cat(r_hmols, 0)

            pooled_r_hmols = torch.unsqueeze(r_hmols, 0)                        # [1, mini_bsz, h]
            pooled_p_hmols = torch.unsqueeze(p_hmols, 0)                        # [1, mini_bsz, h]
            batch_pooled_r_mols.append(pooled_r_hmols)
            batch_pooled_p_mols.append(pooled_p_hmols)

        batch_pooled_r_mols = torch.cat(batch_pooled_r_mols, 0)                 # [bsz, mini_bsz, h]
        batch_pooled_p_mols = torch.cat(batch_pooled_p_mols, 0)                 # [bsz, mini_bsz, h]

        proj_r_mols = self.projection_r(
                                    batch_pooled_r_mols
                                )                                               # [bsz, mini_bsz, h] => [bsz, mini_bsz, d]
        proj_p_mols = self.projection_p(
                                    batch_pooled_p_mols
                                )                                               # [bsz, mini_bsz, h] => [bsz, mini_bsz, d]
                                                                               
        diff = proj_p_mols - proj_r_mols # torch.abs(proj_p_mols - proj_r_mols) # [bsz, mini_bsz, d]
        prod = proj_p_mols * proj_r_mols                                        # [bsz, mini_bsz, d]

        concat = torch.cat([proj_r_mols, proj_p_mols, diff, prod], dim=-1)      # [bsz, mini_bsz, d*4]
        energies = self.output(concat)                                          # [bsz, mini_bsz, 1]
        # debug
        # if torch.isnan(torch.sum(energies, (0, 1, 2))):
        #     print('nan in energies')
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]

class G2E_sep_projBoth_FFout(nn.Module): 
    # separate encoders, linear projections, then a final linear output layer (vs dot product)
    # if no argument is provided for proj_hidden_sizes, there will be no projection
    def __init__(self, args):
        super().__init__()
        self.model_repr = "GraphEBM_sep_projBoth_FFout"
        self.args = args
        self.mol_pool_type = args.mol_pool_type

        if torch.cuda.is_available() and self.args.dataparallel:
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder_p = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type=args.encoder_rnn_type,
                                        h_size=args.encoder_hidden_size,
                                        h_size_inner=args.encoder_inner_hidden_size,
                                        preembed=True if args.preembed_size is not None else False,
                                        preembed_size=args.preembed_size,
                                        depth=args.encoder_depth,
                                        dropout=args.encoder_dropout,
                                        encoder_activation=args.encoder_activation,
                                        atom_pool_type=args.atom_pool_type)

        self.encoder_r = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type=args.encoder_rnn_type,
                                        h_size=args.encoder_hidden_size,
                                        h_size_inner=args.encoder_inner_hidden_size,
                                        preembed=True if args.preembed_size is not None else False,
                                        preembed_size=args.preembed_size,
                                        depth=args.encoder_depth,
                                        dropout=args.encoder_dropout,
                                        encoder_activation=args.encoder_activation,
                                        atom_pool_type=args.atom_pool_type)

        proj_activation = model_utils.get_activation_function(args.proj_activation)
        out_activation = model_utils.get_activation_function(args.out_activation)
        if args.proj_hidden_sizes:
            self.projection_r = self.build_projection(
                proj_activation, args.proj_hidden_sizes, args.proj_dropout, args.encoder_hidden_size
            )
            self.projection_p = self.build_projection(
                proj_activation, args.proj_hidden_sizes, args.proj_dropout, args.encoder_hidden_size
            )
            self.output = self.build_output(
                out_activation, args.out_hidden_sizes, args.out_dropout,
                args.proj_hidden_sizes[-1] * 4
            )
        else:
            self.projection_r, self.projection_p = None, None
            self.output = self.build_output(
                out_activation, args.out_hidden_sizes, args.out_dropout,
                args.encoder_hidden_size * 4
            )            

        if args.do_finetune:
            print("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            print("Setting self.reactant first to True for pretraining")
            self.reactant_first = True
        print("Initializing weights")
        model_utils.initialize_weights(self)
    
    def build_output(
        self,
        activation: nn.Module,
        hidden_sizes: List[int],
        dropout: float,
        input_dim: int,
    ):
        num_layers = len(hidden_sizes)
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
        ffn.extend(
            [
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[-1], 1),
            ]
        )
        return nn.Sequential(*ffn)
    
    def build_projection(
        self,
        activation: nn.Module,
        hidden_sizes: List[int],
        dropout: float,
        input_dim: int,
    ):
        num_layers = len(hidden_sizes)
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
        return nn.Sequential(*ffn)

    def forward(self, batch, probs: Optional[torch.Tensor]=None):
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        graph_tensors, scopes, batch_size = batch
        hatom_r, hmol_r = self.encoder_r(graph_tensors=graph_tensors,
                                   scopes=scopes)
        hatom_p, hmol_p = self.encoder_p(graph_tensors=graph_tensors,
                                   scopes=scopes)

        if self.mol_pool_type == "sum":
            hmol_r = [torch.sum(h, dim=0, keepdim=True) for h in hmol_r]        # list of [n_molecules, h] => list of [1, h]
            hmol_p = [torch.sum(h, dim=0, keepdim=True) for h in hmol_p]
        elif self.mol_pool_type == "mean":
            hmol_r = [torch.mean(h, dim=0, keepdim=True) for h in hmol_r]
            hmol_p = [torch.mean(h, dim=0, keepdim=True) for h in hmol_p]
        else:
            raise NotImplementedError(f"Unsupported mol_pool_type {self.mol_pool_type}! "
                                      f"Please use sum/mean")
        batch_pooled_r_mols = []
        batch_pooled_p_mols = []
    
        mols_per_minibatch = len(hmol_r) // batch_size // self.num_devices  # = (1) r + (mini_bsz) p or (1) p + (mini_bsz) r
        for i in range(batch_size):
            if self.reactant_first:                         # (1) r + mini_bsz p
                r_hmol = hmol_r[i*mols_per_minibatch]                           # [1, h]
                r_hmols = r_hmol.repeat(mols_per_minibatch - 1, 1)              # [mini_bsz, h]
                p_hmols = hmol_p[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                p_hmols = torch.cat(p_hmols, 0)                                 # [mini_bsz, h]
            else:
                p_hmols = hmol_p[i*mols_per_minibatch]        # (1) p + mini_bsz (r)
                p_hmols = p_hmols.repeat(mols_per_minibatch - 1, 1)
                r_hmols = hmol_r[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                r_hmols = torch.cat(r_hmols, 0)

            pooled_r_hmols = torch.unsqueeze(r_hmols, 0)                        # [1, mini_bsz, h]
            pooled_p_hmols = torch.unsqueeze(p_hmols, 0)                        # [1, mini_bsz, h]
            batch_pooled_r_mols.append(pooled_r_hmols)
            batch_pooled_p_mols.append(pooled_p_hmols)

        batch_pooled_r_mols = torch.cat(batch_pooled_r_mols, 0)                 # [bsz, mini_bsz, h]
        batch_pooled_p_mols = torch.cat(batch_pooled_p_mols, 0)                 # [bsz, mini_bsz, h]

        if self.projection_r and self.projection_p:
            proj_r_mols = self.projection_r(
                                        batch_pooled_r_mols
                                    )                                               # [bsz, mini_bsz, h] => [bsz, mini_bsz, d]
            proj_p_mols = self.projection_p(
                                        batch_pooled_p_mols
                                    )                                               # [bsz, mini_bsz, h] => [bsz, mini_bsz, d]
        else:
            proj_r_mols = batch_pooled_r_mols
            proj_p_mols = batch_pooled_p_mols

        diff = proj_p_mols - proj_r_mols # torch.abs(proj_p_mols - proj_r_mols) # [bsz, mini_bsz, d]
        prod = proj_p_mols * proj_r_mols                                        # [bsz, mini_bsz, d]

        concat = torch.cat([proj_r_mols, proj_p_mols, diff, prod], dim=-1)      # [bsz, mini_bsz, d*4]
        energies = self.output(concat)                                          # [bsz, mini_bsz, 1]
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]