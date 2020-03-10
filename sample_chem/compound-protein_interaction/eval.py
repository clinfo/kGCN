import json
import numpy as np
import joblib
data={}
data_mt={}
obj=json.load(open("result_mt/cv.json"))
for fold in obj:
    for itask,task in enumerate(fold):
        for metrics,val in task.items():
            if metrics not in data_mt:
                data_mt[metrics]={}
            if itask not in data_mt[metrics]:
                data_mt[metrics][itask]=[]
            data_mt[metrics][itask].append(val)
data["multitask"]=data_mt
data_mm={}
obj=json.load(open("result_mm/info_cv_mt.json"))
for fold in obj:
    for itask,task in enumerate(fold):
        for metrics,val in task.items():
            if metrics not in data_mm:
                data_mm[metrics]={}
            if itask not in data_mm[metrics]:
                data_mm[metrics][itask]=[]
            data_mm[metrics][itask].append(val)
data["multimodal"]=data_mm

data_st={}
def st(data_st,itask):
    obj=json.load(open("result_st"+str(itask)+"/cv.json"))
    for fold in obj:
        for _,task in enumerate(fold):
            for metrics,val in task.items():
                if metrics not in data_st:
                    data_st[metrics]={}
                if itask not in data_st[metrics]:
                    data_st[metrics][itask]=[]
                data_st[metrics][itask].append(val)

st(data_st,itask=0)
st(data_st,itask=1)
st(data_st,itask=2)
st(data_st,itask=3)

data["singletask"]=data_st



methods=["singletask","multitask","multimodal"]
attr=["auc"]
#attr=["acc","balanced_acc","auc"]
task_names=["MMP12","MMP13","MMP3","MMP9"]
for method in methods:
    for itask,task_name in  enumerate(task_names):
        for m in attr:
            o = data[method][m]
            v=o[itask]
            vv=[method,m,task_name,np.mean(v),np.std(v)]
            print("\t".join(map(str,vv)))
