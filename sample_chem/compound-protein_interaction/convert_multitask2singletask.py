import joblib
import copy
import json
obj=joblib.load("multitask.jbl")
config=json.load(open("config_mt.json"))
print(obj.keys())
for i,name in enumerate(obj["task_names"]):
    print(name)
    o=copy.copy(obj)
    o["label_sparse"]=o["label_sparse"][:,i]
    o["mask_label_sparse"]=o["mask_label_sparse"][:,i]
    o["label_dim"]=1
    o["task_names"]=[name]
    filename="singletask."+str(i)+".jbl"
    joblib.dump(o,filename,compress=3)
    c=copy.copy(config)
    c["save_result_test" ]= "result_st"+str(i)+"/test.csv"
    c["save_result_valid"]= "result_st"+str(i)+"/valid.csv"
    c["save_result_train"]= "result_st"+str(i)+"/train.csv"
    c["save_result_cv"   ]= "result_st"+str(i)+"/cv.json"
    c["save_info_test"   ]= "result_st"+str(i)+"/info_test.json"
    c["save_info_valid"  ]= "result_st"+str(i)+"/info_valid.json"
    c["save_info_train"  ]= "result_st"+str(i)+"/info_train.json"
    c["save_info_cv"     ]= "result_st"+str(i)+"/info_cv.json"
    c["plot_path"        ]= "result_st"+str(i)+"/"
    c["visualize_path"   ]= "viz_st"+str(i)+"/"
    c["load_model"       ]= "model_st"+str(i)+"/model.sample_ckpt"
    c["save_model"       ]= "model_st"+str(i)+"/model.sample_ckpt"
    c["save_model_path"  ]= "model_st"+str(i)+"/"
    c["dataset"          ]= filename
    print("[SAVE] config_st"+str(i)+".json")
    fp = open("config_st"+str(i)+".json", "w")
    json.dump(c, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    
