import numpy as np
import joblib


def make_label_mat(labels,dim=None):
    if dim is None:
        dim=max(labels)+1
    label_mat=np.zeros((len(labels),dim))
    for i,l in enumerate(labels):
        label_mat[i,l]=1
    return label_mat

def make_dummy_adjs(labels):
    adjs=[]
    for i,l in enumerate(labels):
        adjs.append([([],[],(2,2))])
    return adjs

filename="protein.fa"
dataname="data.tsv"
seq_symbol_list=[]
seq_symbol=""
names=[]
labels=[]
for line in open(dataname):
    if len(line.strip())>0:
        arr=line.strip().split("\t")
        names.append(arr[0])
        labels.append(int(arr[1]))
for i,line in enumerate(open(filename)):
    if len(line.strip())==0:
        if i>0:
            seq_symbol_list.append(seq_symbol)
        seq_symbol=""
    elif line[0]==">":
        print(line.strip())
    else:
        seq_symbol+=line.strip()
if len(seq_symbol)>0:
    seq_symbol_list.append(seq_symbol)

###
seq_symbol_list=[[ord(x) - ord("A") for x in seq] for seq in seq_symbol_list]

### save all
max_len_seq=max([len(s) for s in seq_symbol_list])
seq_mat = np.zeros((len(seq_symbol_list), max_len_seq), np.int32)
for i,seq in enumerate(seq_symbol_list):
    seq = list(map(float, seq))
    seq_mat[i, 0:len(seq)] = seq
print(seq_mat.shape)
print(len(names))
label_mat=make_label_mat(labels)
#print(labels)
obj={}
obj["adj"] = make_dummy_adjs(labels)
obj["feature"] = np.zeros((len(labels),2,2))
obj["max_node_num"] = 2
obj["label"] = label_mat
obj["sequence"] = seq_mat
obj["sequence_symbol"] = seq_symbol_list
obj["sequence_length"] = list(map(len, seq_symbol_list))
obj["sequence_symbol_num"] = int(np.max(seq_mat)+1)
v=np.sum(label_mat,axis=0)
class_weight=np.sum(v)/v
obj["class_weight"] = class_weight
print(class_weight)
filename="dataset.jbl"
joblib.dump(obj,filename)

pos_idx=[]
pos_labels=[]
for i,l in enumerate(labels):
    if l>0:
        pos_idx.append(i)
        pos_labels.append(l)
pos_seq_list=[]
pos_seq_mat = np.zeros((len(pos_idx), max_len_seq), np.int32)
for i,j in enumerate(pos_idx):
    pos_seq_mat[i,:]=seq_mat[j, :]
    pos_seq_list.append(seq_symbol_list[j])

obj={}
obj["adj"] = make_dummy_adjs(pos_labels)
obj["feature"] = np.zeros((len(pos_labels),2,2))
obj["max_node_num"] = 2
obj["label"] = make_label_mat(pos_labels,label_mat.shape[1])
obj["sequence"] = pos_seq_mat
obj["sequence_symbol"] = pos_seq_list
obj["sequence_length"] = list(map(len, pos_seq_list))
obj["sequence_symbol_num"] = int(np.max(seq_mat)+1)
filename="pos_dataset.jbl"
joblib.dump(obj,filename)


