import numpy as np
from rdkit import Chem
from typing import Set, Any, List, Union

idxfunc = lambda a: a.GetAtomMapNum() - 1
bond_idx_fn = lambda a, b, mol: mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx()).GetIdx()

# Symbols for different atoms
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LIST)}

MAX_NB = 10
DEGREES = list(range(MAX_NB))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
HYBRIDIZATION_DICT = {hb: i for i, hb in enumerate(HYBRIDIZATION)}

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE)}
VALENCE = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE)}
NUM_Hs = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs)}

CHIRAL_TAG = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
              Chem.rdchem.ChiralType.CHI_UNSPECIFIED]                           # new
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG)}                    # new

RS_TAG = ["R", "S", "None"]                                                     # new
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG)}                            # new

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_DICT = {bond_type: i for i, bond_type in enumerate(BOND_TYPES[1:])}
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_STEREO = [Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREONONE]
BOND_STEREO_DICT = {bond_stereo: i for i, bond_stereo in enumerate(BOND_STEREO)}

BOND_DELTAS = {-3: 0, -2: 1, -1.5: 2, -1: 3, -0.5: 4, 0: 5, 0.5: 6, 1: 7, 1.5: 8, 2: 9, 3: 10}
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]
RXN_CLASSES = list(range(10))

# ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + len(HYBRIDIZATION) \
#             + len(VALENCE) + len(NUM_Hs) + 1
ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + len(HYBRIDIZATION) \
            + len(VALENCE) + len(NUM_Hs) + len(CHIRAL_TAG) + len(RS_TAG) + 2            # new
# BOND_FDIM = 6
BOND_FDIM = 9                                                                           # new
BINARY_FDIM = 5 + BOND_FDIM
INVALID_BOND = -1

ATOM_FDIM = [len(ATOM_LIST), len(DEGREES), len(FORMAL_CHARGE), len(HYBRIDIZATION), len(VALENCE),
             len(NUM_Hs), 2]            # undo one-hot to save space
# BOND_FDIM = [4, 2, 2]                   # undo one-hot to save space
# ATOM_FDIM and BOND_FDIM now contains list of feature dimensions


def onek_encoding_unk(x: Any, allowable_set: Union[List, Set]) -> List:
    """Converts x to one hot encoding.

    Parameters
    ----------
    x: Any,
        An element of any type
    allowable_set: Union[List, Set]
        Allowable element collection
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()


def get_atom_features(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> np.ndarray:
    """Get atom features.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    if atom.GetSymbol() == '*':
        symbol = onek_encoding_unk(atom.GetSymbol(), ATOM_LIST)
        if use_rxn_class:
            padding = [0] * (ATOM_FDIM + len(RXN_CLASSES) - len(symbol))
        else:
            padding = [0] * (ATOM_FDIM - len(symbol))
        feature_array = symbol + padding
        return np.array(feature_array)

    if use_rxn_class:
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST) +
                        onek_encoding_unk(atom.GetDegree(), DEGREES) +
                        onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE) +
                        onek_encoding_unk(atom.GetHybridization(), HYBRIDIZATION) +
                        onek_encoding_unk(atom.GetTotalValence(), VALENCE) +
                        onek_encoding_unk(atom.GetTotalNumHs(), NUM_Hs) +
                        [float(atom.GetIsAromatic())] + onek_encoding_unk(rxn_class, RXN_CLASSES))

    else:
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST) +
                        onek_encoding_unk(atom.GetDegree(), DEGREES) +
                        onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE) +
                        onek_encoding_unk(atom.GetHybridization(), HYBRIDIZATION) +
                        onek_encoding_unk(atom.GetTotalValence(), VALENCE) +
                        onek_encoding_unk(atom.GetTotalNumHs(), NUM_Hs) +
                        [float(atom.GetIsAromatic())])


def get_atom_features_sparse(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> List[int]:
    """Get atom features as sparse idx.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    feature_array = []
    symbol = atom.GetSymbol()
    symbol_id = ATOM_DICT.get(symbol, ATOM_DICT["unk"])
    feature_array.append(symbol_id)

    if symbol in ["*", "unk"]:
        # padding = [999999999] * 7 if use_rxn_class else [999999999] * 6
        padding = [999999999] * ATOM_FDIM if use_rxn_class else [999999999] * (ATOM_FDIM - 1) # new
        feature_array.extend(padding)

    else:
        degree_id = atom.GetDegree()
        if degree_id not in DEGREES:
            degree_id = 9
        formal_charge_id = FC_DICT.get(atom.GetFormalCharge(), 4)
        hybridization_id = HYBRIDIZATION_DICT.get(atom.GetHybridization(), 4)
        valence_id = VALENCE_DICT.get(atom.GetTotalValence(), 6)
        num_h_id = NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4)
        chiral_tag_id = CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2)                 # new

        rs_tag = atom.GetPropsAsDict().get("_CIPCode", "None")                      # new
        rs_tag_id = RS_TAG_DICT.get(rs_tag, 2)                                      # new

        is_aromatic = int(atom.GetIsAromatic())
        # feature_array.extend([degree_id, formal_charge_id, hybridization_id,
        #                       valence_id, num_h_id, is_aromatic])
        feature_array.extend([degree_id, formal_charge_id, hybridization_id,        # new
                              valence_id, num_h_id, chiral_tag_id, rs_tag_id, is_aromatic])

        if use_rxn_class:
            feature_array.append(rxn_class)

    return feature_array


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    # bond_features = [float(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    # bond_features.extend([float(bond.GetIsConjugated()), float(bond.IsInRing())])
    bond_features = [int(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bs = bond.GetStereo()                                                               # new
    bond_features.extend([int(bs == bond_stereo) for bond_stereo in BOND_STEREO])       # new
    bond_features.extend([int(bond.GetIsConjugated()), int(bond.IsInRing())])
    # bond_features = np.array(bond_features, dtype=np.float32)
    return bond_features


def get_bond_features_sparse(bond: Chem.Bond) -> List[int]:
    """Get bond features as sparse idx.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bs = bond.GetStereo()                                                               # new
    # bond_features = [BOND_DICT[bt], int(bond.GetIsConjugated()), int(bond.IsInRing())]
    bond_features = [BOND_DICT[bt], int(bond.GetIsConjugated()), int(bond.IsInRing())   # new
                    BOND_STEREO_DICT[bs]]  # new

    return bond_features
