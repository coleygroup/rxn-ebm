import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
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
        # TODO: add dropout here
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
                                  h_size=self.h_size,
                                  depth=self.depth)

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
        atom_scope, bond_scope = scopes

        # embed graph, note that for directed graph, fess[any, 0:2] = u, v
        hnode = fnode.clone()
        fmess1 = hnode.index_select(index=fmess[:, 0].long(), dim=0)
        fmess2 = fmess[:, 2:].clone()
        hmess = torch.cat([fmess1, fmess2], dim=-1)

        # encode
        hatom, _ = self.encoder(hnode, hmess, agraph, bgraph, mask=None)

        hmol = []
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
        return hatom, hmol


class G2E(nn.Module):
    def __init__(self, args, encoder_hidden_size, encoder_depth, **kwargs):
        super().__init__()
        self.model_repr = "GraphEBM"
        self.args = args

        if torch.cuda.is_available():
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type="gru",
                                        h_size=encoder_hidden_size,
                                        depth=encoder_depth)
        self.output = nn.Linear(encoder_hidden_size * 4, 1)
        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True
        logging.info("Initializing weights")
        model_utils.initialize_weights(self)

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

        hmol = [torch.sum(h, dim=0, keepdim=True) for h in hmol]        # list of [n_molecules, h] => list of [1, h]
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

            diff = torch.abs(p_hmols - r_hmols)                                 # [mini_bsz, h]
            prod = r_hmols * p_hmols                                            # [mini_bsz, h]

            pooled_hmols = torch.cat([r_hmols, p_hmols, diff, prod], 1)         # [mini_bsz, h*4]
            pooled_hmols = torch.unsqueeze(pooled_hmols, 0)                     # [1, mini_bsz, h*4]

            batch_pooled_hmols.append(pooled_hmols)

        batch_pooled_hmols = torch.cat(batch_pooled_hmols, 0)                   # [bsz, mini_bsz, h*4]
        energies = self.output(batch_pooled_hmols)                              # [bsz, mini_bsz, 1]
        # logging.info("-------energies-------")
        # logging.info(energies)
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]

class G2E_sep(nn.Module): # separate encoders
    def __init__(self, args, encoder_hidden_size, encoder_depth, **kwargs):
        super().__init__()
        self.model_repr = "GraphEBM_sep"
        self.args = args

        if torch.cuda.is_available():
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder_p = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type="gru",
                                        h_size=encoder_hidden_size,
                                        depth=encoder_depth)
        
        self.encoder_r = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type="gru",
                                        h_size=encoder_hidden_size,
                                        depth=encoder_depth)

        self.output = nn.Linear(encoder_hidden_size * 4, 1)
        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True
        logging.info("Initializing weights")
        model_utils.initialize_weights(self)

    def forward(self, batch, probs: Optional[torch.Tensor]=None):
        """
        batch: a N x K x 1 tensor of N training samples
            each sample contains a positive rxn on the first column,
            and K-1 negative rxns on all subsequent columns
        """
        graph_tensors, scopes, batch_size = batch

        # how to get product graph tensors & scopes?
        hatom_r, hmol_r = self.encoder_r(graph_tensors=graph_tensors,
                                   scopes=scopes)
        hatom_p, hmol_p = self.encoder_p(graph_tensors=graph_tensors,
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

        hmol_r = [torch.sum(h, dim=0, keepdim=True) for h in hmol_r]        # list of [n_molecules, h] => list of [1, h]
        hmol_p = [torch.sum(h, dim=0, keepdim=True) for h in hmol_p]        # list of [n_molecules, h] => list of [1, h]
        # logging.info([h.size() for h in hmol])

        batch_pooled_hmols = []
    
        mols_per_minibatch = len(hmol_r) // batch_size // self.num_devices  # = (1) r + (mini_bsz) p or (1) p + (mini_bsz) r
        # assert mols_per_minibatch == self.args.minibatch_size + 1, \
        #     f"calculated minibatch size: {mols_per_minibatch-1}, given in args: {self.args.minibatch_size}"

        # logging.info(f"{len(hmol)}, {batch_size}, {mols_per_minibatch}")
        for i in range(batch_size):
            if self.reactant_first:                         # (1) r + mini_bsz p
                r_hmol = hmol_r[i*mols_per_minibatch]                             # [1, h]
                r_hmols = r_hmol.repeat(mols_per_minibatch - 1, 1)              # [mini_bsz, h]
                p_hmols = hmol_p[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                p_hmols = torch.cat(p_hmols, 0)                                 # [mini_bsz, h]
            else:
                p_hmols = hmol_p[i*mols_per_minibatch]        # (1) p + mini_bsz (r)
                p_hmols = p_hmols.repeat(mols_per_minibatch - 1, 1)
                r_hmols = hmol_r[(i*mols_per_minibatch+1):(i+1)*mols_per_minibatch]
                r_hmols = torch.cat(r_hmols, 0)

            diff = torch.abs(p_hmols - r_hmols)                                 # [mini_bsz, h]
            prod = r_hmols * p_hmols                                            # [mini_bsz, h]

            pooled_hmols = torch.cat([r_hmols, p_hmols, diff, prod], 1)         # [mini_bsz, h*4]
            pooled_hmols = torch.unsqueeze(pooled_hmols, 0)                     # [1, mini_bsz, h*4]

            batch_pooled_hmols.append(pooled_hmols)

        batch_pooled_hmols = torch.cat(batch_pooled_hmols, 0)                   # [bsz, mini_bsz, h*4]
        energies = self.output(batch_pooled_hmols)                              # [bsz, mini_bsz, 1]
        # logging.info("-------energies-------")
        # logging.info(energies)
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]


class G2E_projR(nn.Module):
    def __init__(self, args, encoder_hidden_size, encoder_depth, proj_hidden_sizes, proj_activation, proj_dropout, **kwargs):
        super().__init__()
        self.model_repr = "GraphEBM_projR"
        self.args = args

        if torch.cuda.is_available():
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type="gru",
                                        h_size=encoder_hidden_size,
                                        depth=encoder_depth)
        
        self.projection = self.build_projection(
            proj_activation, proj_hidden_sizes, proj_dropout,
            encoder_hidden_size, # output_dim = encoder_hidden_size
        )
        
        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True
        logging.info("Initializing weights")
        model_utils.initialize_weights(self)
    
    def build_projection(
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
                activation.
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[-1], input_dim), # output_dim = encoder_hidden_size = input_dim
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

        hmol = [torch.sum(h, dim=0, keepdim=True) for h in hmol]        # list of [n_molecules, h] => list of [1, h]
        # logging.info([h.size() for h in hmol])

        batch_pooled_r_mols = []
        batch_pooled_p_mols = []
    
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

            pooled_r_hmols = torch.unsqueeze(r_hmols, 0)                        # [1, mini_bsz, h]
            pooled_p_hmols = torch.unsqueeze(p_hmols, 0)                        # [1, mini_bsz, h]
            batch_pooled_r_mols.append(pooled_r_hmols)
            batch_pooled_p_mols.append(pooled_p_hmols)

        batch_pooled_r_mols = torch.cat(batch_pooled_r_mols, 0)                 # [bsz, mini_bsz, h]
        batch_pooled_p_mols = torch.cat(batch_pooled_p_mols, 0)                 # [bsz, mini_bsz, h]

        proj_pooled_r_mols = self.projection(
                                    batched_pooled_r_mols
                                ).unsqueeze(dim=-1)                             # [bsz, mini_bsz, h] => [bsz, mini_bsz, h] => [bsz, mini_bsz, h, 1]
        
        energies = torch.matmul(
                        batch_pooled_p_mols.tranpose(),
                        proj_pooled_r_mols
                    ).squeeze(dim=-1)                                           # [bsz, mini_bsz, 1, h] x [bsz, mini_bsz, h, 1] => [bsz, mini_bsz, 1]
        # logging.info("-------energies-------")
        # logging.info(energies)
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]


class G2E_projBoth(nn.Module):
    def __init__(self, args, encoder_hidden_size, encoder_depth, proj_hidden_sizes, proj_activation, proj_dropout, **kwargs):
        super().__init__()
        self.model_repr = "GraphEBM_projBoth"
        self.args = args

        if torch.cuda.is_available():
            self.num_devices = torch.cuda.device_count()
        else:
            self.num_devices = 1

        self.encoder = GraphFeatEncoder(n_atom_feat=sum(ATOM_FDIM),
                                        n_bond_feat=BOND_FDIM,
                                        rnn_type="gru",
                                        h_size=encoder_hidden_size,
                                        depth=encoder_depth)
        
        proj_activation = model_utils.get_activation_function(proj_activation)
        self.projection_r = self.build_projection(
            proj_activation, proj_hidden_sizes, proj_dropout, encoder_hidden_size # output_dim = proj_hidden_sizes[-1]
        )
        self.projection_p = self.build_projection(
            proj_activation, proj_hidden_sizes, proj_dropout, encoder_hidden_size # output_dim = proj_hidden_sizes[-1]
        )
        
        if args.do_finetune:
            logging.info("Setting self.reactant first to False for finetuning")
            self.reactant_first = False
        else:
            logging.info("Setting self.reactant first to True for pretraining")
            self.reactant_first = True
        logging.info("Initializing weights")
        model_utils.initialize_weights(self)
    
    def build_projection(
        self,
        activation: nn.Module,
        hidden_sizes: List[int],
        dropout: float,
        input_dim: int,
    ):
        num_layers = len(hidden_sizes)
        ffn = [nn.Linear(input_dim, hidden_sizes[0])] # could add nn.Dropout(dropout) if we wish
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

        hmol = [torch.sum(h, dim=0, keepdim=True) for h in hmol]        # list of [n_molecules, h] => list of [1, h]
        # logging.info([h.size() for h in hmol])

        batch_pooled_r_mols = []
        batch_pooled_p_mols = []
    
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
        energies = torch.matmul(
                        torch.transpose(proj_pooled_p_mols, 2, 3),
                        proj_pooled_r_mols
                    ).squeeze(dim=-1)                                           # [bsz, mini_bsz, 1, d] x [bsz, mini_bsz, d, 1] => [bsz, mini_bsz, 1]
        # logging.info("-------energies-------")
        # logging.info(energies)
        return energies.squeeze(dim=-1)                                         # [bsz, mini_bsz]