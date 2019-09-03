import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
import joblib

def read_label_file(csvfile):
    csv = pd.read_csv(csvfile,header=None)
    label = np.array(csv.values)
    # Convert nan to mask
    mask_label=np.zeros_like(label,dtype=np.float32)
    mask_label[~np.isnan(label)]=1
    label[np.isnan(label)]=0
    return label,mask_label

def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
        )
    parser.add_argument(
        '--label', default=None,type=str,
        help='help'
    )
    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='help'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    if args.label is None:
        print("[ERROR] --label is required")
        quit()
                
    if args.output is None:
        print("[ERROR] --output is required")
        quit()

    label_data,label_mask=read_label_file(args.label)

    # joblib output
    obj = {}
    obj["label"]        = np.asarray(label_data)
    obj["mask_label"]   = np.asarray(label_mask)

    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)

if __name__ == "__main__":
    main()
