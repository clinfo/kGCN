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
        '--modality', default=None,type=str,
        help='Modality joblib file'
    )
    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='Output dataset joblib file'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    if args.dataset is None:
        print("[ERROR] --dataset is required")
        quit()

    if args.modality is None:
        print("[ERROR] --moadlity is required")
        quit()

    if args.output is None:
        print("[ERROR] --output is required")
        quit()

    # joblib output
    obj = joblib.load(args.dataset)
    modality = joblib.load(args.modality)
    obj.update(modality)
    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)

if __name__ == "__main__":
    main()
