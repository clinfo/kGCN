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

def config_copy(src,dest,key,i):
    data=src[key]
    path, ext = os.path.splitext(data)
    dest[key]=path+"."+str(i)+ext

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
    idx=np.array(list(range(data_num)))
    np.random.seed(1234)
    np.random.shuffle(idx)
    fold=0
    kfold = KFold(n_splits=args.fold)
    for train_idx, test_idx in kfold.split(np.zeros(data_num,)):
        ## setting dataset
        data_train,data_test=kgcn.data_util.split_jbl_obj(obj)
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
        fold+=1




