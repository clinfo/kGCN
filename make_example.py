import numpy as np
import joblib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--multilabel', action='store_true')
parser.add_argument('--data', type=str,
        default="example_data",
        nargs='?',
        help='data path')
parser.add_argument('--output', type=str,
        default=None,
        nargs='?',
        help='save jbl file')
args=parser.parse_args()

print("[LOAD] "+args.data+"/adj.txt")
n=None
mat=[]
list_mat=[]
for i,line in enumerate(open(args.data+"/adj.txt")):
    sline=line.strip()
    arr=sline.split(",")
    if sline!="":
        vec=list(map(int,arr))
        if n is None:
            n=len(vec)
        mat.append(vec)
    else:
        list_mat.append(np.array(mat))
        mat=[]
if len(mat)>0:
    list_mat.append(np.array(mat))
mm=np.array(list_mat)
max_node_num = np.max([mat.shape[0] for mat in mm])

print("[LOAD] "+args.data+"/feature.txt")
feature=[]
list_feature=[]
for i,line in enumerate(open(args.data+"/feature.txt")):
    sline=line.strip()
    arr=sline.split(",")
    if sline!="":
        vec=list(map(float,arr))
        feature.append(vec)
    else:
        list_feature.append(feature)
        feature=[]
if len(feature)>0:
    list_feature.append(feature)
ff=np.array(list_feature)

if args.multilabel:
    print("[LOAD] "+args.data+"/multi_label.txt")
    labels=[]
    mask_label=[]
    for i,line in enumerate(open(args.data+"/multi_label.txt")):
        arr=line.strip().split(",")
        vec=[]
        mask_vec=[]
        for el in arr:
            if el.strip()=="":
                vec.append(0)
                mask_vec.append(0)
            else:
                vec.append(float(el))
                mask_vec.append(1)

        labels.append(vec)
        mask_label.append(mask_vec)
    ll=np.array(labels)
    ml=np.array(mask_label)
else:

    print("[LOAD] "+args.data+"/label.txt")
    labels=[]
    mask_label=[]
    for i,line in enumerate(open(args.data+"/label.txt")):
        arr=line.strip().split(",")
        vec=[]
        mask_vec=[]
        for el in arr:
            if el.strip()=="":
                vec.append(0)
                mask_vec.append(0)
            else:
                vec.append(float(el))
                mask_vec.append(1)
        labels.append(vec)
        mask_label.append(mask_vec)
    ll=np.array(labels)
    ml=np.array(mask_label)

obj={"feature":ff,
"dense_adj":mm,
"label":ll,
"mask_label":ml,
"max_node_num":max_node_num}


if(os.path.exists(args.data+"/seq.txt")):
    print("[LOAD] "+args.data+"/seq.txt")
    seqs=[]
    for i,line in enumerate(open(args.data+"/seq.txt")):
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

if args.output is None:
    if args.multilabel:
        filename="example_jbl/sample_multitask.jbl"
    else:
        filename="example_jbl/sample.jbl"
else:
        filename=args.output
print("[SAVE] "+filename)
joblib.dump(obj, filename)


