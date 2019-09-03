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
        '--label', default=None,type=str,
        help='Input label joblib file'
    )
    parser.add_argument(
        '--adjacent', default=None,type=str,
        help='Input adjacent joblib file'
    )    
    parser.add_argument(
        '--atom_feature', default=None,type=str,
        help='Input atom features joblib file'
    )
    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='Output dataset joblibfile'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    if args.label is None:
        print("[ERROR] --label is required")
        quit()
                
    if args.adjacent is None:
        print("[ERROR] --adjacent is required")
        quit()
                
    if args.atom_feature is None:
        print("[ERROR] --atom_feature is required")
        quit()
                
    if args.output is None:
        print("[ERROR] --output is required")
        quit()

    # joblib output
    obj = joblib.load(args.label)
    obj.update(joblib.load(args.adjacent))
    obj.update(joblib.load(args.atom_feature))
    obj["max_node_num"] = obj["atom_num_limit"]
    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)

if __name__ == "__main__":
    main()
