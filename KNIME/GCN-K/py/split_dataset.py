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
        '--ratio', default=None,type=str,
        help='Ratio of 1st dataset'
    )
    parser.add_argument(
        '-o1', '--output1', default=None,type=str,
        help='Output 1st dataset joblib file'
    )
    parser.add_argument(
        '-o2', '--output2', default=None,type=str,
        help='Output 2nd dataset joblib file'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    if args.dataset is None:
        print("[ERROR] --dataset is required")
        quit()
                
    if args.ratio is None:
        print("[ERROR] --ratio is required")
        quit()

    if args.output1 is None:
        print("[ERROR] --output1 is required")
        quit()

    if args.output2 is None:
        print("[ERROR] --output2 is required")
        quit()

    ratio = float(args.ratio)
    if (ratio < 0.0) or (1.0 < ratio):
        print("[ERROR] invalid ratio value")
        quit()
    
    obj = joblib.load(args.dataset)
    adjs = obj["adj"]
    nmol = len([ adj for adj in adjs if adj is not None])
    nmol1 = int(nmol * ratio)

    if 0 < nmol1:
        for i in range(len(adjs)):
            if adjs[i] is not None:
                nmol1 -= 1
            if nmol1 == 0:
                nmol1 = i + 1
                break

    keys = ['label', 'feature', 'mask_label', 'adj',
        'vector_modal', 'profeat', 'dragon', 'chemical_fp','mol_info']


    obj1 = obj.copy()
    obj2 = obj.copy()
    for key in keys:
        if key =="mol_info":
            obj1[key] = {}
            obj2[key] = {}
            for mol_key in obj[key].keys():
                obj1[key][mol_key]=obj[key][mol_key][:nmol1]
                obj2[key][mol_key]=obj[key][mol_key][nmol1:]
        elif key not in obj.keys():
            continue
        else:
            obj1[key] = obj[key][:nmol1]
            obj2[key] = obj[key][nmol1:]

    filename=args.output1
    print("[SAVE] "+filename)
    joblib.dump(obj1, filename)

    filename=args.output2
    print("[SAVE] "+filename)
    joblib.dump(obj2, filename)

if __name__ == "__main__":
    main()
