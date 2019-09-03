import argparse
import os
import numpy as np
import joblib

def get_parser():
    parser = argparse.ArgumentParser(
            description='description',
            usage='usage'
        )
    parser.add_argument(
        '--profeat', default=None,type=str,
        help='profeat csv file'
    )
    parser.add_argument(
        '--sequence', default=None,type=str,
        help='sequence csv file'
    )
    parser.add_argument(
        '-o', '--output', default=None,type=str,
        help='output dataset jbl file'
    )
    return parser.parse_args()

def main():
    args = get_parser()

    if args.output is None:
        print("[ERROR] --output is required")
        quit()

    # joblib output
    obj = {}

    if args.profeat is not None and (os.path.exists(args.profeat)):
        print("[LOAD] " + args.profeat)
        profeat=[]
        for i,line in enumerate(open(args.profeat).readlines()):
            arr=line.strip().split(",")
            vec=list(map(float,arr))
            profeat.append(vec)
        obj["profeat"]=np.array(profeat)
        del profeat

    if args.sequence is not None and (os.path.exists(args.sequence)):
        print("[LOAD] " + args.sequence)
        seqs=[]
        for i,line in enumerate(open(args.sequence)):
            arr=line.strip().split(",")
            vec=list(map(float,arr))
            seqs.append(vec)
        max_len_seq=max(map(len,seqs))
        seq_mat=np.zeros((len(seqs),max_len_seq),np.int32)
        for i,seq in enumerate(seqs):
            seq_mat[i,0:len(seq)]=seq
        obj["sequence"]=seq_mat
        obj["sequence_length"]=list(map(len,seqs))
        obj["sequence_symbol_num"]=np.max(seq_mat)+1

    filename=args.output
    print("[SAVE] "+filename)
    joblib.dump(obj, filename)

if __name__ == "__main__":
    main()
