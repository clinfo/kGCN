import joblib
import sys
import json
import re
import argparse
import numpy as np
import copy
import os
from sklearn.model_selection import KFold
import kgcn.data_util
#split_jbl_obj(obj,train_idx,test_idx,label_list_flag=False,index_list_flag=False)

def config_copy(args,src,dest,key,i):
    if key in src:
        data=src[key]
        #path, ext = os.path.splitext(data)
        dest[key]=args.cv_path+"/fold"+str(i)+"/"+data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
            required=True,
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
    parser.add_argument('--fold', type=int,
            default=5,
            help='#fold')
    parser.add_argument('--seed', type=int,
            default=1234,
            help='seed')
    parser.add_argument('--inhibit_shuffle', 
            action='store_true',
            help='without shuffle')

    args=parser.parse_args()

    # config
    config={}#get_default_config()
    print("[LOAD] ",args.config)
    fp = open(args.config, 'r')
    config.update(json.load(fp))

    cv=args.cv_path
    os.makedirs(cv,exist_ok=True)
    # option
    if args.dataset is not None:
        config["dataset"]=args.dataset
    dataset_name=config["dataset"]

    
    print("[LOAD]",dataset_name)
    obj=joblib.load(dataset_name)
    print("input keys:",obj.keys())
    data_num=kgcn.data_util.get_data_num_jbl_obj(obj)
    print("#data:",data_num)
    ###
    np.random.seed(args.seed)
    i=0
    cv_data_info=[]
    kfold = KFold(n_splits=args.fold,shuffle=not args.inhibit_shuffle)
    for train_idx, test_idx in kfold.split(np.zeros(data_num,)):
        ## setting dataset
        data_train,data_test=kgcn.data_util.split_jbl_obj(obj,train_idx,test_idx)
        name, ext = os.path.splitext( os.path.basename(dataset_name) )
        train_filename=cv+"/"+name+".train_"+str(i)+".jbl"
        test_filename=cv+"/"+name+".test_"+str(i)+".jbl"
        print("[SAVE]",train_filename)
        joblib.dump(data_train, train_filename)
        print("[SAVE]",test_filename)
        joblib.dump(data_test, test_filename)
        #
        config_fold=copy.deepcopy(config)
        config_fold["dataset"]=train_filename
        config_fold["dataset_test"]=test_filename
        #
        config_copy(args,config,config_fold,"save_result_test",i)
        config_copy(args,config,config_fold,"save_result_valid",i)
        config_copy(args,config,config_fold,"save_result_train",i)
        config_copy(args,config,config_fold,"save_result_cv",i)
        config_copy(args,config,config_fold,"save_info_test",i)
        config_copy(args,config,config_fold,"save_info_valid",i)
        config_copy(args,config,config_fold,"save_info_train",i)
        config_copy(args,config,config_fold,"save_info_cv",i)
        config_copy(args,config,config_fold,"save_model",i)
        config_copy(args,config,config_fold,"load_model",i)
        config_copy(args,config,config_fold,"plot",i)
        config_copy(args,config,config_fold,"save_model_path",i)
        #
        name, ext = os.path.splitext( os.path.basename(args.config) )
        config_filename=cv+"/"+name+"."+str(i)+".json"
        print("[SAVE]",config_filename)
        fp=open(config_filename,"w")
        json.dump(config_fold,fp, indent=4)
        cv_data_info.append({"train_index":train_idx.tolist(),"test_index":test_idx.tolist()})
        i+=1
    config_filename=cv+"/cv.json"
    print("[SAVE]",config_filename)
    fp=open(config_filename,"w")
    json.dump(cv_data_info,fp, indent=4)




