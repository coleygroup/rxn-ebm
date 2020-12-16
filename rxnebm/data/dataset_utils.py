import logging
import networkx as nx
import re
import torch
from rdkit import Chem
from rxnebm.data.chem_utils import ATOM_FDIM, BOND_FDIM, get_atom_features_sparse, get_bond_features
from rxnebm.data.rxn_graphs import RxnGraph
from typing import List, Tuple


def get_graph_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    rxn_graph = RxnGraph(reac_mol=mol)
    return rxn_graph


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist]) + 1
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.tensor(alist, dtype=torch.long)


def get_features_per_graph(smi: str, use_rxn_class: bool):
    atom_features = []
    bond_features = []

    graph = get_graph_from_smiles(smi).reac_mol

    mol = graph.mol
    assert mol.GetNumAtoms() == len(graph.G_dir)

    G = nx.convert_node_labels_to_integers(graph.G_dir, first_label=0)

    for v, attr in G.nodes(data="label"):
        atom_feat = get_atom_features_sparse(mol.GetAtomWithIdx(v),
                                             use_rxn_class=use_rxn_class,
                                             rxn_class=graph.rxn_class)
        atom_features.append(atom_feat)

    for u, v, attr in G.edges(data='label'):
        bond_feat = get_bond_features(mol.GetBondBetweenAtoms(u, v))
        bond_feat = [u, v] + bond_feat
        bond_features.append(bond_feat)

    return graph, G, atom_features, bond_features


def densify(features: List[List[int]], FDIM: List[int]) -> List[List[int]]:
    one_hot_features = []
    for feature in features:
        one_hot_feature = [0] * sum(FDIM)
        for i, idx in enumerate(feature):
            if idx == 9999:         # padding
                continue
            one_hot_feature[idx+sum(FDIM[:i])] = 1

        one_hot_features.append(one_hot_feature)

    return one_hot_features


def get_graph_features(batch_graphs_and_features: List[Tuple], directed: bool = True,
                       use_rxn_class: bool = False) -> Tuple[Tuple, Tuple[List, List]]:
    if directed:
        padded_features = get_atom_features_sparse(Chem.Atom("*"), use_rxn_class=use_rxn_class, rxn_class=0)
        padded_features = densify([padded_features], ATOM_FDIM)
        fnode = padded_features
        fmess = [[0, 0] + [0] * BOND_FDIM]
        agraph, bgraph = [[]], [[]]
        unique_bonds = {(0, 0)}

        atom_scope, bond_scope = [], []
        edge_dict = {}

        for bid, graphs_and_features in enumerate(batch_graphs_and_features):
            graph, G, atom_features, bond_features = graphs_and_features
            # densify on the fly temporarily, TODO: to be fully converted into embedding based
            atom_features = densify(atom_features, ATOM_FDIM)
            # bond_features = densify(bond_features, BOND_FDIM)

            atom_offset = len(fnode)
            bond_offset = len(unique_bonds)
            atom_scope.append(graph.update_atom_scope(atom_offset))
            bond_scope.append(graph.update_bond_scope(bond_offset))

            # node iteration is reduced to an extend
            fnode.extend(atom_features)
            agraph.extend([[] for _ in range(len(atom_features))])

            # first edge iteration
            for bond_feat in bond_features:
                u, v = bond_feat[:2]
                u_adj = u + atom_offset
                v_adj = v + atom_offset
                bond = tuple(sorted([u_adj, v_adj]))
                unique_bonds.add(bond)

                # do not use assign for list or it'll be passed by reference
                mess_vec = [u_adj, v_adj] + bond_feat[2:]

                fmess.append(mess_vec)
                edge_dict[(u_adj, v_adj)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v_adj].append(eid)
                bgraph.append([])

            # second edge iteration (after edge_dict is updated fully)
            for bond_feat in bond_features:
                u, v = bond_feat[:2]
                u_adj = u + atom_offset
                v_adj = v + atom_offset

                eid = edge_dict[(u_adj, v_adj)]
                for w in G.predecessors(u):
                    if w == v:
                        continue
                    w_adj = w + atom_offset
                    bgraph[eid].append(edge_dict[(w_adj, u_adj)])

        fnode = torch.tensor(fnode, dtype=torch.float)
        fmess = torch.tensor(fmess, dtype=torch.float)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        graph_tensors = (fnode, fmess, agraph, bgraph, None)
        scopes = (atom_scope, bond_scope)

    else:
        raise NotImplementedError("Zhengkai will get the undirected graph if needed")

    return graph_tensors, scopes


def graph_collate_fn_builder(device, debug: bool):
    """Creates an 'collate_fn' closure to be passed to DataLoader, for graph encoders"""
    def collate_fn(data):           # list of bsz (list of K smiles)
        """The actual collate_fn"""
        batch_graphs_and_features = []
        batch_masks = []
        batch_idxs = []

        # each graphs_and_features is a minibatch
        # each masks is a minibatch too
        for graphs_and_features, masks, idx in data:
            batch_graphs_and_features.extend(graphs_and_features)
            batch_masks.append(masks)
            batch_idxs.append(idx)

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

        return (graph_tensors, scopes, batch_size), batch_masks, batch_idxs

    return collate_fn


def smi_tokenizer(smile):
    """
    Tokenize a SMILES molecule or reaction
    taken from https://github.com/pschwllr/MolecularTransformer
    """
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smile)]
    assert smile == ''.join(tokens)
    return tokens


def seq_collate_fn_builder(device, debug: bool):
    """Creates an 'collate_fn' closure to be passed to DataLoader, for graph encoders"""
    def collate_fn(data):           # list of bsz (list of K smiles)
        """The actual collate_fn"""

        '''
        batch_smis = []
        batch_masks = []
        for rxn_smiles, masks in data:
            r_smi = rxn_smiles[0].split(">>")[0]
            p_smis = [rxn_smi.split(">>")[-1] for rxn_smi in rxn_smiles]

            batch_smis.append(r_smi)
            batch_smis.extend(p_smis)
            batch_masks.append([bool(smi) for smi in p_smis])
        '''

        batch_graphs_and_features = []
        batch_masks = []

        # each graphs_and_features is a minibatch
        # each masks is a minibatch too
        for graphs_and_features, masks in data:
            batch_graphs_and_features.extend(graphs_and_features)
            batch_masks.append(masks)

        batch_size = len(data)
        batch_masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)
        graph_tensors, scopes = get_graph_features(batch_graphs_and_features=batch_graphs_and_features,
                                                   use_rxn_class=False)
        graph_tensors = [tensor.to(device) for tensor in graph_tensors[:4]]
        graph_tensors.append(None)

        if debug:
            # logging.info("-------graph tensors-------")
            # logging.info(graph_tensors)
            logging.info("-------scopes-------")
            logging.info(scopes)
            logging.info("-------batch_masks-------")
            logging.info(batch_masks)

        return (graph_tensors, scopes, batch_size), batch_masks

    return collate_fn