import numpy as np
import joblib
import os
import sys
ENABLE_MULTI_LABEL=False

filename="./example_jbl/synthetic.jbl"
if len(sys.argv)>1:
    filename=sys.argv[1]
print("[LOAD]",filename)
obj=joblib.load(filename)

features=obj["feature"]
mm=obj["dense_adj"]

## 
node_labels=[]
mapping={}
for fv in features:
    nl=[]
    for f in fv:
        k=tuple(map(int,f))
        if k not in mapping:
            mapping[k]=len(mapping)
        nl.append(mapping[k])
    node_labels.append(nl)
maxN=np.max([len(nl) for nl in node_labels])
print(maxN)
for nl in node_labels:
    nlv=[0]*maxN
    l=len(nl)
    nlv[:l]=nl[:]
###
adjs=[]
for mat in mm:
    es=[]
    vs=[]
    n=len(mat)
    for i,v in enumerate(mat):
        for j,el in enumerate(v):
            if int(el)!=0:
                es.append([i,j])
                vs.append(1)
    adjs.append([np.array(es),np.array(vs),[n,n]])

del obj["feature"]
del obj["dense_adj"]

obj["adj"]=adjs
obj["node"]=node_labels
##
print("[SAVE] example_jbl/synthetic_sparse.jbl")
joblib.dump(obj, 'example_jbl/synthetic_sparse.jbl')


