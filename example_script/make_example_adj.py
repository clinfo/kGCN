import numpy as np
import joblib
import os
import sys

# usage: python make_example_adj.py ./example_data/adj?.txt

ENABLE_MULTI_LABEL=False

multi_adjs=[]
for arg in sys.argv[1:]:
    filename=arg
    print("[LOAD]",filename)
    n=None
    mat=[]
    list_mat=[]
    for i,line in enumerate(open(filename)):
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
    multi_adjs.append(list_mat)
max_node_num = np.max([[mat.shape[0] for mat in mm] for mm in multi_adjs])

## check adjs
a=np.unique([len(list_mat) for list_mat in multi_adjs])
if len(a)!=1:
    print("[ERROR] miss-matched the number of adj. matrices")
num_graph=a[0]
##
num_adj=len(multi_adjs)
data_adjs=[[None for _ in range(num_adj)]for _ in range(num_graph)]
for i in range(num_graph):
    for j in range(num_adj):
        a=multi_adjs[j][i]
        data_adjs[i][j]=a
##
print("[LOAD] example_data/feature.txt")
feature=[]
list_feature=[]
for i,line in enumerate(open("example_data/feature.txt")):
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

if ENABLE_MULTI_LABEL:
    print("[LOAD] example_data/multi_label.txt")
    labels=[]
    mask_label=[]
    for i,line in enumerate(open("example_data/multi_label.txt")):
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

    print("[LOAD] example_data/label.txt")
    labels=[]
    mask_label=[]
    for i,line in enumerate(open("example_data/label.txt")):
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
"multi_dense_adj":data_adjs,
"num_adj":num_adj,
"label":ll,
"mask_label":ml,
"max_node_num":max_node_num}


if(os.path.exists("example_data/seq.txt")):
    print("[LOAD] example_data/seq.txt")
    seqs=[]
    for i,line in enumerate(open("example_data/seq.txt")):
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

if ENABLE_MULTI_LABEL:
    print("[SAVE] example_data/sample_multitask.jbl")
    joblib.dump(obj, 'example_data/sample_multitask.jbl')
else:
    print("[SAVE]", 'example_jbl/multiadj.jbl')
    joblib.dump(obj, 'example_jbl/multiadj.jbl')


