import logging
import networkx as nx
import numpy as np
import re
import torch
from rdkit import Chem
from rxnebm.data.chem_utils import ATOM_FDIM, BOND_FDIM, get_atom_features_sparse, get_bond_features
from rxnebm.data.rxn_graphs import RxnGraph
from typing import Dict, List, Tuple


def get_graph_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    rxn_graph = RxnGraph(reac_mol=mol)
    return rxn_graph


def get_features_per_graph(smi: str, use_rxn_class: bool):
    atom_features = []
    bond_features = []
    edge_dict = {}

    graph = get_graph_from_smiles(smi).reac_mol

    mol = graph.mol
    assert mol.GetNumAtoms() == len(graph.G_dir)

    G = nx.convert_node_labels_to_integers(graph.G_dir, first_label=0)

    # node iteration to get sparse atom features
    for v, attr in G.nodes(data="label"):
        atom_feat = get_atom_features_sparse(mol.GetAtomWithIdx(v),
                                             use_rxn_class=use_rxn_class,
                                             rxn_class=graph.rxn_class)
        atom_features.append(atom_feat)

    a_graphs = [[] for _ in range(len(atom_features))]

    # edge iteration to get (dense) bond features
    for u, v, attr in G.edges(data='label'):
        bond_feat = get_bond_features(mol.GetBondBetweenAtoms(u, v))
        bond_feat = [u, v] + bond_feat
        bond_features.append(bond_feat)

        eid = len(edge_dict)
        edge_dict[(u, v)] = eid
        a_graphs[v].append(eid)

    b_graphs = [[] for _ in range(len(bond_features))]

    # second edge iteration to get neighboring edges (after edge_dict is updated fully)
    for bond_feat in bond_features:
        u, v = bond_feat[:2]
        eid = edge_dict[(u, v)]

        for w in G.predecessors(u):
            if not w == v:
                b_graphs[eid].append(edge_dict[(w, u)])

    # padding
    for a_graph in a_graphs:
        while len(a_graph) < 7:
            a_graph.append(9999)

    for b_graph in b_graphs:
        while len(b_graph) < 7:
            b_graph.append(9999)

    a_scopes = np.array(graph.atom_scope, dtype=np.int32)
    a_scopes_lens = a_scopes.shape[0]
    b_scopes = np.array(graph.bond_scope, dtype=np.int32)
    b_scopes_lens = b_scopes.shape[0]
    a_features = np.array(atom_features, dtype=np.int32)
    a_features_lens = a_features.shape[0]
    b_features = np.array(bond_features, dtype=np.int32)
    b_features_lens = b_features.shape[0]
    a_graphs = np.array(a_graphs, dtype=np.int32)
    b_graphs = np.array(b_graphs, dtype=np.int32)

    return a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, \
        a_features, a_features_lens, b_features, b_features_lens, a_graphs, b_graphs


def densify(features: List[List[int]], FDIM: List[int]) -> List[List[int]]:
    one_hot_features = []
    for feature in features:
        one_hot_feature = [0] * sum(FDIM)
        for i, idx in enumerate(feature):
            if idx == 9999:         # padding and unknown
                continue
            one_hot_feature[idx+sum(FDIM[:i])] = 1

        one_hot_features.append(one_hot_feature)

    return one_hot_features


def get_graph_features(batch_graphs_and_features: List[Tuple], directed: bool = True,
                       use_rxn_class: bool = False) -> Tuple[Tuple, Tuple[List, List]]:
    if directed:
        padded_features = get_atom_features_sparse(Chem.Atom("*"), use_rxn_class=use_rxn_class, rxn_class=0)
        fnode = [np.array(padded_features)]
        fmess = [np.zeros(shape=[1, 2+BOND_FDIM], dtype=np.int32)]
        agraph = [np.zeros(shape=[1, 7], dtype=np.int32)]
        bgraph = [np.zeros(shape=[1, 7], dtype=np.int32)]

        n_unique_bonds = 1
        edge_offset = 1

        atom_scope, bond_scope = [], []

        for bid, graphs_and_features in enumerate(batch_graphs_and_features):
            a_scope, b_scope, atom_features, bond_features, a_graph, b_graph = graphs_and_features

            atom_offset = len(fnode)
            bond_offset = n_unique_bonds
            n_unique_bonds += int(bond_features.shape[0] / 2)              # This should be correct?

            a_scope[:, 0] += atom_offset
            b_scope[:, 0] += bond_offset
            atom_scope.append(a_scope)
            bond_scope.append(b_scope)

            # node iteration is reduced to an extend
            fnode.extend(atom_features)

            # edge iteration is reduced to an append
            bond_features[:, :2] += atom_offset
            fmess.append(bond_features)

            a_graph += edge_offset
            a_graph[a_graph >= 9999] = 0            # resetting padding edge to point towards edge 0
            agraph.append(a_graph)

            b_graph += edge_offset
            b_graph[b_graph >= 9999] = 0            # resetting padding edge to point towards edge 0
            bgraph.append(b_graph)

            edge_offset += bond_features.shape[0]

        # densification
        fnode = np.stack(fnode, axis=0)
        fnode_one_hot = np.zeros([fnode.shape[0], sum(ATOM_FDIM)], dtype=np.float32)

        for i in range(len(ATOM_FDIM) - 1):
            fnode[:, i+1:] += ATOM_FDIM[i]

        for i, feat in enumerate(fnode):            # Looks vectorizable?
            fnode_one_hot[i, feat[feat < 9999]] = 1

        fnode = torch.from_numpy(fnode_one_hot)
        fmess = torch.from_numpy(np.concatenate(fmess, axis=0))
        agraph = torch.from_numpy(np.concatenate(agraph, axis=0)).long()
        bgraph = torch.from_numpy(np.concatenate(bgraph, axis=0)).long()

        graph_tensors = (fnode, fmess, agraph, bgraph, None)
        scopes = (atom_scope, bond_scope)

    else:
        raise NotImplementedError("Zhengkai will get the undirected graph if needed")

    return graph_tensors, scopes


def graph_collate_fn_builder(device, debug: bool):
    """Creates an 'collate_fn' closure to be passed to DataLoader, for graph encoders"""
    def collate_fn(data):           # list of bsz (list of K)
        """The actual collate_fn"""
        batch_graphs_and_features = []
        batch_masks = []
        batch_idxs = []
        batch_probs = []

        # each graphs_and_features is a minibatch
        # each masks is a minibatch too
        for graphs_and_features, masks, idx, probs in data:
            batch_graphs_and_features.extend(graphs_and_features)
            batch_masks.append(masks)
            batch_idxs.append(idx)
            batch_probs.append(probs)

        batch_size = len(data)
        batch_masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)

        graph_tensors, scopes = get_graph_features(batch_graphs_and_features=batch_graphs_and_features,
                                                   use_rxn_class=False)

        graph_tensors = [tensor.to(device) for tensor in graph_tensors[:4]]
        graph_tensors.append(None)      # for compatibility

        if debug:
            logging.info("-------graph tensors-------")
            logging.info(graph_tensors)
            logging.info("-------scopes-------")
            logging.info(scopes)
            logging.info("-------batch_masks-------")
            logging.info(batch_masks)

        return (graph_tensors, scopes, batch_size), batch_masks, batch_idxs, batch_probs

    return collate_fn


def smi_tokenizer(smiles: str) -> List[str]:
    """
    Tokenize a SMILES molecule or reaction
    taken from https://github.com/pschwllr/MolecularTransformer
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]
    assert smiles == "".join(tokens), \
        f"Tokenization error. SMILES: {smiles}, tokenized: {tokens}"
    return tokens


def get_seq_features_per_minibatch(minibatch_smiles: List[str],
                                   vocab: Dict[str, int],
                                   max_seq_len: int) -> Tuple[List[List[int]], List[int]]:
    minibatch_token_ids = []
    minibatch_lens = []

    for smi in minibatch_smiles:
        tokens = smi_tokenizer(smi)
        token_ids = [vocab["_CLS"]]
        token_ids.extend([vocab[token] if token in vocab else vocab["_UNK"]
                          for token in tokens])
        token_ids = token_ids[:max_seq_len-1]
        token_ids.append(vocab["_SEP"])
        seq_len = len(token_ids)

        # padding
        while len(token_ids) < max_seq_len:
            token_ids.append(vocab["_PAD"])

        minibatch_token_ids.append(token_ids)
        minibatch_lens.append(seq_len)

    return minibatch_token_ids, minibatch_lens


def seq_collate_fn_builder(device, vocab: Dict[str, int], max_seq_len: int = 512, debug: bool = False):
    """Creates an 'collate_fn' closure to be passed to DataLoader, for transformer encoders"""
    def collate_fn(data):           # list of bsz (list of K)
        """The actual collate_fn"""
        batch_token_ids = []
        batch_lens = []
        batch_masks = []
        batch_idxs = []
        batch_probs = []

        for rxn_smiles_with_negatives, masks, idx, probs in data:
            minibatch_token_ids, minibatch_lens = get_seq_features_per_minibatch(
                rxn_smiles_with_negatives, vocab=vocab, max_seq_len=max_seq_len)
            batch_token_ids.extend(minibatch_token_ids)
            batch_lens.extend(minibatch_lens)

            batch_masks.append(masks)
            batch_idxs.append(idx)
            batch_probs.append(probs)

        batch_size = len(data)
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.long, device=device)
        batch_masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)
        batch_probs = torch.tensor(batch_probs, dtype=torch.float, device=device)

        if debug:
            logging.info("-------token_id tensors-------")
            logging.info(batch_token_ids)

        return (batch_token_ids, batch_lens, batch_size), batch_masks, batch_idxs, batch_probs

    return collate_fn
