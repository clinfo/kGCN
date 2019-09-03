import argparse
import os

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
        '--dataset', default=None,type=str,
        help='Input dataset joblib file'
    )
    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='Output dataset joblibfile'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    if args.dataset is None:
        print("[ERROR] --dataset is required")
        quit()
                
    if args.output is None:
        print("[ERROR] --output is required")
        quit()

    keys = ['label', 'feature', 'mask_label', 'adj',
        'vector_modal', 'profeat', 'dragon', 'chemical_fp','mol_info']

    obj = joblib.load(args.dataset)
    adjs = obj["adj"]
    for key in keys:
        if key =="mol_info":
            for mol_key in obj[key].keys():
                obj[key][mol_key] = [ val for (adj, val) in zip(adjs,obj[key][mol_key])
                             if adj is not None]
                obj[key][mol_key] = np.array(obj[key][mol_key])
        elif key not in obj.keys():
            continue
        else:
            obj[key] = [ val for (adj, val) in zip(adjs, obj[key])
                             if adj is not None]
            obj[key] = np.array(obj[key])

    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)

if __name__ == "__main__":
    main()
