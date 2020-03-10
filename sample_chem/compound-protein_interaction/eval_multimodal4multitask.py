import json
import numpy as np
import joblib
import sklearn.metrics

tasks=[]
for line in open("multimodal_data_index.csv"):
    row=line.strip().split(",")
    tasks.append(row[3])
obj=json.load(open("result_mm/info_cv.json"))
data=[]
task_data_pred={}
task_data_label={}
for k,fold in enumerate(obj):
    task_data_pred[k]={}
    task_data_label[k]={}
    for i,idx in enumerate(fold["test_data_idx"]):
        pred=fold["prediction_data"][i]
        l=fold["test_labels"][i]
        t=tasks[idx]
        data.append([idx,t,pred[1],l[1]])
        if t not in task_data_pred[k]:
            task_data_pred[k][t]=[]
            task_data_label[k][t]=[]
        task_data_pred[k][t].append(pred[1])
        task_data_label[k][t].append(l[1])

print("#data:",len(data))
#for in sorted(data)
task_names=["sample/MMP12_HUMAN","sample/MMP13_HUMAN","sample/MMP13_HUMAN","sample/MMP9_HUMAN"]
out_task_names=["MMP12","MMP13","MMP3","MMP9"]
cv_data=[]
for fold in task_data_label.keys():
    cv_data_task=[]
    for t in task_names:
        y_true=np.array(task_data_label[fold][t])
        y_score=np.array(task_data_pred[fold][t])
        y_pred=np.zeros_like(y_score)
        y_pred[y_score>0.5]=1
        #print(y_true,y_score)
        acc=sklearn.metrics.accuracy_score(y_true,y_pred)
        auc=sklearn.metrics.roc_auc_score(y_true,y_score)
        balanced_acc=sklearn.metrics.balanced_accuracy_score(y_true,y_pred)
        el={
                "acc":acc,
                "auc":auc,
                "balanced_acc":balanced_acc,
                }
        print(t,acc,auc)
        cv_data_task.append(el)
    cv_data.append(cv_data_task)
filename="result_mm/info_cv_mt.json"
fp=open(filename,"w")
json.dump(cv_data,fp)

