import numpy as np
import joblib
import os

ENABLE_MULTI_LABEL=True

print("[LOAD] example_data/adj.txt")
n=None
mat=[]
list_mat=[]
for i,line in enumerate(open("example_data/adj.txt")):
    sline=line.strip()
    arr=sline.split(",")
    if sline!="":
        vec=list(map(int,arr))
        if n is None:
            n=len(vec)
        mat.append(vec)
    else:
        list_mat.append(mat)
        mat=[]
if len(mat)>0:
    list_mat.append(mat)
mm=np.array(list_mat)
max_node_num = np.max([mat.shape[0] for mat in mm])


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

print("[LOAD] example_data/node_label.txt")
labels=[]
mask_label=[]
mat=[]
mask_mat=[]
for i,line in enumerate(open("example_data/node_label.txt")):
    sline=line.strip()
    arr=sline.split(",")
    if sline != "":
        vec=[]
        mask_vec=[]
        for el in arr:
            if el.strip()=="":
                vec.append(0)
                mask_vec.append(0)
            else:
                vec.append(float(el))
                mask_vec.append(1)
        mat.append(vec)
        mask_mat.append(mask_vec)
    else:
        labels.append(mat)
        mask_label.append(mask_mat)
        mat=[]
        mask_mat=[]
if len(mat)>0:
    labels.append(mat)
    mask_label.append(mask_mat)
print(labels)
ll=np.array(labels)
ml=np.array(mask_label)

obj={"feature":ff,
"dense_adj":mm,
"node_label":ll,
"mask_node_label":ml,
"max_node_num":max_node_num}

print("[SAVE] example_jbl/sample_node_label.jbl")
joblib.dump(obj, 'example_jbl/sample_node_label.jbl')


