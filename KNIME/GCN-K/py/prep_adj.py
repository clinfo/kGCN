import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
import joblib

def dense_to_sparse(dense):
    from scipy.sparse import coo_matrix
    coo=coo_matrix(dense)
    sh=coo.shape
    val=coo.data
    sp=list(zip(coo.row,coo.col))
    return (np.array(sp),np.array(val,dtype=np.float32),np.array(sh))

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

    adj_list = []
    for index, mol in enumerate(mol_obj_list):
        #Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SANITIZE_ADJUSTHS)
        if mol is None:    
            adj_list.append(None)
            continue
        # Create a adjacency matrix
        mol_adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        row_num=len(mol_adj)
        adj = np.array(mol_adj, dtype=np.int)
        for i in range(row_num):  # Set diagonal elements to 1, fill others with the adjacency matrix from RDkit
            adj[i][i] = int(1)

        adj_list.append(dense_to_sparse(adj))

    # joblib output
    obj["adj"] = np.asarray(adj_list)
    print(obj)
    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)

if __name__ == "__main__":
    main()
