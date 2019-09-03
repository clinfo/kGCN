import argparse
import os

import numpy as np
from rdkit import Chem
import joblib

def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
        )
    parser.add_argument(
        '--mol_info', default=None,type=str,
        help='help'
    )
    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='help'
    )
    return parser.parse_args()

# Get the information from atom
def atom_features(atom, bool_id_feat=False, explicit_H=False):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C',
                'N',
                'O',
                'S',
                'F',
                'Si',
                'P',
                'Cl',
                'Br',
                'Mg',
                'Na',
                'Ca',
                'Fe',
                'As',
                'Al',
                'I',
                'B',
                'V',
                'K',
                'Tl',
                'Yb',
                'Sb',
                'Sn',
                'Ag',
                'Pd',
                'Co',
                'Se',
                'Ti',
                'Zn',
                'H',  # H?
                'Li',
                'Ge',
                'Cu',
                'Au',
                'Ni',
                'Cd',
                'In',
                'Mn',
                'Zr',
                'Cr',
                'Pt',
                'Hg',
                'Pb',
                'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
    return np.array(results)

# Convert to one-hot expression
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def main():
    args = get_parser()

    if args.mol_info is None:
        print("[ERROR] --mol_info is required")
        quit()
                
    if args.output is None:
        print("[ERROR] --output is required")
        quit()

    obj = joblib.load(args.mol_info)
    mol_obj_list  = obj["mol_info"]["obj_list"]
    mol_name_list = obj["mol_info"]["name_list"]
    atom_num_limit = obj["atom_num_limit"]

    feature_list = []
    for index, mol in enumerate(mol_obj_list):
        if mol is None:
            feature_list.append(None)
            continue
        # Create a feature matrix
        feature = [atom_features(atom) for atom in mol.GetAtoms()]
        for _ in range(atom_num_limit - len(feature)):
            feature.append(np.zeros(len(feature[0]), dtype=np.int))

        feature_list.append(feature)

    # joblib output
    obj = {}
    obj["feature"] = np.asarray(feature_list)
    obj["atom_num_limit"] = atom_num_limit

    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)

if __name__ == "__main__":
    main()
