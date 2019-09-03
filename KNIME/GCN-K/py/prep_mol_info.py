import argparse
import os
import glob

import numpy as np
import pandas as pd
from rdkit import Chem
import joblib

def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
        )
    parser.add_argument(
        '--sdf', default=None,type=str,
        help='help'
    )
    parser.add_argument(
        '-a', '--atom_num_limit', default=None, required=True, type=int,
        help='help'
    )
    parser.add_argument(
        '-o', '--output', default="dataset.jbl",type=str,
        help='help'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    mol_name_list=[]
    mol_obj_list=[]
    if args.sdf is not None:
        # Extract mol information from the file
        filename=args.sdf
        if not os.path.exists(filename):
            print("[PASS] not found:",filename)
        sdf_filename = filename.replace("//", "/")
        mol_obj_list=[mol for mol in Chem.SDMolSupplier(sdf_filename)]
    else:
        print("[ERROR] --sdf is required")
        quit()

    for index, mol in enumerate(mol_obj_list):
        #Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SANITIZE_ADJUSTHS)
        if mol is None:
            continue
        # Skip the compound whose total number of atoms is larger than "atom_num_limit"
        if args.atom_num_limit is not None and  mol.GetNumAtoms() > args.atom_num_limit:
            # do not remove mol because labels are read from CSV later 
            mol_obj_list[index] = None
            mol_name_list.append("")
            continue
        # Get mol. name
        name = mol.GetProp("_Name")
        mol_name_list.append(name)

    # joblib output
    obj = {}
    mol_info = {"obj_list": mol_obj_list, "name_list": mol_name_list}
    obj["mol_info"] = mol_info
    obj["atom_num_limit"] = args.atom_num_limit

    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)


if __name__ == "__main__":
    main()
