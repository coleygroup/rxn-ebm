# try:
#     from crem_updated import crem # has SQL hack, but it doesn't really boost performance
# except ImportError:
from crem import crem

import os
import pickle
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesWriter
from tqdm import tqdm


def mol_smi_from_mol_pickle(
    input_pickle_filename: str = "50k_mol_smis",
    output_smi_filename: str = "50k_mol_smis",
    root: Optional[Union[str, bytes, os.PathLike]] = None,
):
    """helper function to generate .smi file from .pickle file of molecular SMILES
    needed to create custom fragment database if one so desires (not recommended)
    """
    if root is None:
        root = Path(__file__).resolve().parents[2] / "data" / "cleaned_data"

    with open(root / f"{input_filename}.pickle", "rb") as handle:
        mol_smis = pickle.load(handle)

    with SmilesWriter(root / f"{output_filename}.smi") as writer:
        for mol_smi in tqdm(mol_smis):
            mol = Chem.MolFromSmiles(mol_smi)
            writer.write(mol)


def gen_crem_negs(
    num_neg: int,
    max_size: int,
    radius: int,
    frag_db_filename: Union[str, bytes, os.PathLike] = "replacements02_sa2.db",
    rxn_smi_file_prefix: Union[str, bytes, os.PathLike] = "50k_clean_rxnsmi_noreagent",
    dataset_name: Optional[str] = "50k",
    root: Optional[Union[str, bytes, os.PathLike]] = None,
    phases: Optional[Union[str, List[str]]] = None,
    ncores: Optional[int] = 1,
):
    """
    Assumes that there is only 1 product in every reaction!

    Parameters
    ----------
    frag_db_filename : Union[str, bytes, os.PathLike] (Default = 'replacements02_sa2.db')
        filename of the context-fragment database. Highly recommended to use the pre-built ones
        from CReM's author. must be placed in the folder specified by 'root', which is, by default,
        'path/to/rxn-ebm/data/cleaned_data/
    ncores : Optional[int] (Default = 1)
        number of cores to use. Highly recommended to use 1. Depending on machine, overhead of
        parallelising can outweigh the speed-up, slowing down the process (up to 2x slower!!)

    """
    if phases is None:
        phases = ["train", "valid", "test"]
    if root is None:
        root = Path(__file__).resolve().parents[2] / "data" / "cleaned_data"
    if (
        root
        / f"{dataset_name}_neg{num_neg}_rad{radius}_maxsize{max_size}_mutprodsmis.pickle"
    ).exists():
        print("The mutprodsmis file already exists!")
        return
    else:
        print("mutprodsmis file not found. Generating crem negatvies...")
        print("This will take a while!! (~9-12 hours for 150 negs/rxn on USPTO_50k...)")

    all_mut_prod_smi = {}
    insufficient = {}
    for phase in phases:
        with open(root / f"{rxn_smi_file_prefix}_{phase}.pickle", "rb") as handle:
            rxn_smi_phase = pickle.load(handle)

        for i, rxn_smi in enumerate(tqdm(rxn_smi_phase[:20])):
            prod_smi = rxn_smi.split(">>")[-1]
            prod_mol = Chem.MolFromSmiles(prod_smi)

            this_rxn_mut = []
            j = 0
            for j, mut_prod_smi in enumerate(
                crem.mutate_mol(
                    prod_mol,
                    db_name=str(root / frag_db_filename),
                    radius=radius,
                    max_size=max_size,
                    return_mol=False,
                    ncores=ncores,
                )
            ):
                this_rxn_mut.append(mut_prod_smi)
                j += 1
                if j > num_neg - 1:
                    break

            all_mut_prod_smi[prod_smi] = this_rxn_mut

            if j < num_neg - 1:
                # print(f'At index {i}, {j}<{num_neg}')
                insufficient[rxn_smi] = j

            if i % 2500 == 0:  # checkpoint
                with open(
                    root
                    / f"{dataset_name}_neg{num_neg}_rad{radius}_maxsize{max_size}_{phase}_mutprodsmis.pickle",
                    "wb",
                ) as handle:
                    pickle.dump(all_mut_prod_smi, handle, pickle.HIGHEST_PROTOCOL)
                with open(
                    root
                    / f"{dataset_name}_neg{num_neg}_rad{radius}_maxsize{max_size}_{phase}_insufficient.pickle",
                    "wb",
                ) as handle:
                    pickle.dump(insufficient, handle, pickle.HIGHEST_PROTOCOL)

        with open(
            root
            / f"{dataset_name}_neg{num_neg}_rad{radius}_maxsize{max_size}_{phase}_mutprodsmis.pickle",
            "wb",
        ) as handle:
            pickle.dump(all_mut_prod_smi, handle, pickle.HIGHEST_PROTOCOL)
        with open(
            root
            / f"{dataset_name}_neg{num_neg}_rad{radius}_maxsize{max_size}_{phase}_insufficient.pickle",
            "wb",
        ) as handle:
            pickle.dump(insufficient, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # to create custom fragment database:
    # mol_smi_from_pickle('50k_mol_smis', '50k_mol_smis') then run crem_create_frag_db.sh from command prompt

    gen_crem_negs(
        num_neg=150, max_size=3, radius=2, frag_db_filename="replacements02_sa2.db"
    )
