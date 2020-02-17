
import glob
import os
import re
import json
obj=json.load(open("result_mt/info_cv.json"))
cv_data={}
for k,fold in enumerate(obj):
    ls=[]
    preds=[]
    scores=[]
    for l,pred in zip(fold["test_labels"],fold["prediction_data"]):
        l=[int(e) for e in l]
        score=[score for score in pred]
        pred=[1 if score>=0.5 else 0 for score in pred]
        ls.append(l)
        preds.append(pred)
        scores.append(score)
    cv_data[k]=list(zip(fold["test_data_idx"],ls,preds,scores))

filelist=glob.glob("./viz_mt/*.jbl")
data={}
for f in filelist:
    #./viz_mm/fold3_2351_task_0_inactive_all_scaling.jbl
    name=os.path.basename(f)
    name,_=os.path.splitext(name)
    arr=name.split("_")
    fold_s=arr[0]
    task_s=arr[3]
    idx=int(arr[1])
    m=re.match(r"fold([0-9]+)",fold_s)
    if m:
        fold=int(m.group(1))
        if fold not in data:
            data[fold]=[]
        data[fold].append((idx,int(task_s),name))
all_data=[]
for k,v in data.items():
    for el in v:
        idx=el[0]
        task=el[1]
        name=el[2]
        org_idx_pair=cv_data[k][idx]
        org_idx=org_idx_pair[0]
        l=org_idx_pair[1][task]
        pred=org_idx_pair[2][task]
        score=org_idx_pair[3][task]
        #all_data.append((org_idx,k,el[0],el[1],el[2]))
        all_data.append((org_idx,task,l,pred,score,k,idx,name))
    print(k,len(v))

index_data={}
for line in open("multimodal_data_index.csv"):
    arr=line.strip().split(",")
    index_data[int(arr[0])]=arr[2]
print(index_data)

print(len(all_data))
fp=open("summary_viz_mt.tsv","w")
s1="\t".join(["compound ID","task ID","SMILES","label","prediction","score","fold ID","fold index","visualization filename"])
fp.write(s1)
fp.write("\n")
for el in sorted(all_data):
    print(el)
    s1="\t".join(map(str,el[0:2]))
    s2="\t".join(map(str,el[2:]))
    smi=index_data[el[0]]
    fp.write(s1+"\t"+smi+"\t"+s2)
    fp.write("\n")

