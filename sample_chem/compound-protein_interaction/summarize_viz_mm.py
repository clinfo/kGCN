
import glob
import os
import re
import json
obj=json.load(open("result_mm/info_cv.json"))
cv_data={}
for k,fold in enumerate(obj):
    ls=[int(l[1]) for l in fold["test_labels"]]
    preds=[1 if pred[1]>0.5 else 0 for pred in fold["prediction_data"]]
    scores=[pred[1] for pred in fold["prediction_data"]]
    cv_data[k]=list(zip(fold["test_data_idx"],ls,preds,scores))

filelist=glob.glob("./viz_mm/*.jbl")
data={}
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
        data[fold].append((idx,name))
all_data=[]
for k,v in data.items():
    for el in v:
        idx=el[0]
        org_idx_pair=cv_data[k][idx]
        org_idx,l,pred,score=org_idx_pair
        all_data.append((org_idx,l,pred,score,k,el[0],el[1]))
    print(k,len(v))

index_data=[]
for line in open("multimodal_data_index.csv"):
    arr=line.strip().split(",")
    index_data.append(arr)

print(len(index_data))
print(len(all_data))
#['5252', '3', 'CC(C)C(C(=O)NO)N(Cc1cccnc1)S(=O)(=O)c1ccc(F)cc1', 'sample/MMP9_HUMAN'] (15027, 1, 1, 3, 3004, 'fold3_3004_task_0_active_all_scaling')
fp=open("summary_viz_mm.tsv","w")
s1="\t".join(["compound ID","task ID","SMILES","task"])
s2="\t".join(["index","label","prediction","score","fold ID","fold index","visualization filename"])
fp.write(s1+"\t"+s2)
fp.write("\n")
for index_el,el in zip(index_data,sorted(all_data)):
    print(index_el,el)
    s="\t".join(index_el)+"\t"+"\t".join(map(str,el))
    fp.write(s)
    fp.write("\n")
