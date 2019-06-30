import joblib
import sys
import json
import re
import argparse
import numpy as np
import copy
import os

def config_copy(src,dest,key,i):
    data=src[key]
    path, ext = os.path.splitext(data)
    dest[key]=path+"."+str(i)+ext

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
            default=None,
            nargs='?',
            help='config json file')
    parser.add_argument('--save-config',
            default=None,
            nargs='?',
            help='save config json file')
    parser.add_argument('--no-config',
            action='store_true',
            help='use default setting')
    parser.add_argument('--model', type=str,
            default=None,
            help='model')
    parser.add_argument('--dataset', type=str,
            default=None,
            help='dataset')
    parser.add_argument('--cv_path', type=str,
            default="cv",
            help='dataset')

    args=parser.parse_args()
    # config
    config={}#get_default_config()
    if args.config is None:
        pass
        parser.print_help()
        quit()
    else:
        print("[LOAD] ",args.config)
        fp = open(args.config, 'r')
        config.update(json.load(fp))
    cv=args.cv_path
    if not os.path.exists(cv):
        os.makedirs(cv)
    # option
    if args.dataset is not None:
        config["dataset"]=args.dataset
    dataset_name=config["dataset"]

    graph_id=0
    obj=joblib.load(dataset_name)
    print(obj.keys())
    data_num=0
    if "label" in obj:
        v=obj["label"]
        print("#label     :",len(v))
        data_num=len(v)
    else:
        print("[ERROR] No label data")
        quit()
    ###
    # checking
    if "adj" in obj:
        v=obj["adj"]
        print("#adj       :",len(v))
    if "graph_name" in obj:
        v=obj["graph_name"]
        print("#graph_name:",len(v))
    if "node" in obj:
        v=obj["node"]
        print("#node      :",len(v))
    ###
    idx=np.array(list(range(data_num)))
    np.random.seed(1234)
    np.random.shuffle(idx)
    fold=5
    for i in range(fold):
        ## splitting dataset
        train_idx=[]
        test_idx=[]
        for j in range(fold):
            n=int(data_num*j*1.0/fold)
            if j+1==fold:
                m=data_num
            else:
                m=int(data_num*(j+1)*1.0/fold)
            if i==j:
                test_idx.extend(idx[n:m])
            else:
                train_idx.extend(idx[n:m])
        #print(len(test_idx),"//",len(train_idx))
        ## setting dataset
        fold_dataset_test={}
        fold_dataset_train={}
        direct_copy_keys=["max_node_num","mol_info"]
        for key,val in obj.items():
            if key not in direct_copy_keys:
                print(key,": split")
                o=np.array(obj[key])
                fold_dataset_test[key]=o[test_idx]
                fold_dataset_train[key]=o[train_idx]
            else:
                print(key,": direct copy")
                fold_dataset_test[key]=obj[key]
                fold_dataset_train[key]=obj[key]
        name, ext = os.path.splitext( os.path.basename(dataset_name) )
        train_filename=cv+"/"+name+".train_"+str(i)+".jbl"
        test_filename=cv+"/"+name+".test_"+str(i)+".jbl"
        print("[SAVE]",train_filename)
        joblib.dump(fold_dataset_train, train_filename)
        print("[SAVE]",test_filename)
        joblib.dump(fold_dataset_test, test_filename)
        #
        config_fold=copy.deepcopy(config)
        config_fold["dataset"]=train_filename
        config_fold["dataset_test"]=test_filename
        #
        config_copy(config,config_fold,"save_result_test",i)
        config_copy(config,config_fold,"save_result_train",i)
        config_copy(config,config_fold,"save_model",i)
        config_copy(config,config_fold,"load_model",i)
        #
        path=cv+"/"+config["save_model_path"]+"_"+str(i)
        if not os.path.exists(path):
            os.makedirs(path)
        config_fold["save_model_path"]=path
        #
        name, ext = os.path.splitext( os.path.basename(args.config) )
        config_filename=cv+"/"+name+"."+str(i)+".json"
        print("[SAVE]",config_filename)
        fp=open(config_filename,"w")
        json.dump(config_fold,fp, indent=4)




