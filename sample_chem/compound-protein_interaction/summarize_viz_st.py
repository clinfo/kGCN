
import glob
import os
import re
import json
cv_data={}
tasks=[0,1,2,3]
for st in tasks:
    cv_data[st]={}
    print("[LOAD] result_st"+str(st)+"/info_cv.json")
    obj=json.load(open("result_st"+str(st)+"/info_cv.json"))
    for k,fold in enumerate(obj):
        ls=[int(l[0]) for l in fold["test_labels"]]
        if isinstance(fold["prediction_data"][0][0],list):
            preds=[1 if float(pred[0][0]) > 0.5 else 0 for pred in fold["prediction_data"]]
            score=[float(pred[0][0]) for pred in fold["prediction_data"]]
        else:
            preds=[1 if float(pred[0]) > 0.5 else 0 for pred in fold["prediction_data"]]
            score=[float(pred[0]) for pred in fold["prediction_data"]]
        cv_data[st][k]=list(zip(fold["test_data_idx"],ls,preds,score))

data={}
for st in tasks:
    filelist=glob.glob("./viz_st"+str(st)+"/*.jbl")
    for f in filelist:
        #./viz_mm/fold3_2351_task_0_inactive_all_scaling.jbl
        name=os.path.basename(f)
        name,_=os.path.splitext(name)
        arr=name.split("_")
        fold_s=arr[0]
        idx=int(arr[1])
        m=re.match(r"fold([0-9]+)",fold_s)
        if m:
            fold=int(m.group(1))
            if fold not in data:
                data[fold]=[]
            data[fold].append((idx,st,name))

all_data=[]
for k,v in data.items():
    for el in v:
        idx=el[0]
        task=el[1]
        name=el[2]
        org_idx_pair=cv_data[task][k][idx]
        org_idx=org_idx_pair[0]
        l=org_idx_pair[1]
        pred=org_idx_pair[2]
        score=org_idx_pair[3]
        #all_data.append((org_idx,k,el[0],el[1],el[2]))
        all_data.append((org_idx,task,l,pred,score,k,idx,name))
    print(k,len(v))

index_data={}
for line in open("multimodal_data_index.csv"):
    arr=line.strip().split(",")
    index_data[int(arr[0])]=arr[2]
#print(index_data)

print(len(all_data))
fp=open("summary_viz_st.tsv","w")
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

