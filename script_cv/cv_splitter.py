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

def info_cv_splitter(info_filename):
    cv_info=json.load(open(info_filename))
    test_data_idx=[el["test_data_idx"] for el in cv_info]
    n=len(test_data_idx)
    for i, test in enumerate(test_data_idx):
        train=[]
        for j in range(n):
            if i!=j:
                train+=test_data_idx[j]
        yield np.array(train), np.array(test)


def config_copy(args,src,dest,key,i):
    if key in src:
        data=src[key]
        #path, ext = os.path.splitext(data)
        dest[key]=args.cv_path+"/fold"+str(i)+"/"+data

def main():
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
    parser.add_argument('--prohibit_shuffle', 
            action='store_true',
            help='without shuffle')
    parser.add_argument('--without_config', 
            action='store_true',
            help='without config output')
    parser.add_argument('--without_train', 
            action='store_true',
            help='without train data output')
    parser.add_argument('--without_test', 
            action='store_true',
            help='without test data output')
    parser.add_argument('--use_info', 
            action='store_true',
            help='using cv_info to split data')

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
    if args.use_info:
        info_name=config["save_info_cv"]
        splitter=info_cv_splitter(info_name)
    else:
        kfold = KFold(n_splits=args.fold,shuffle=not args.prohibit_shuffle)
        splitter=kfold.split(np.zeros(data_num,))


    for train_idx, test_idx in splitter:
        ## setting dataset
        data_train,data_test=kgcn.data_util.split_jbl_obj(obj,train_idx,test_idx)
        name, ext = os.path.splitext( os.path.basename(dataset_name) )
        train_filename=cv+"/"+name+".train_"+str(i)+".jbl"
        test_filename=cv+"/"+name+".test_"+str(i)+".jbl"
        if not args.without_train:
            print("[SAVE]",train_filename)
            joblib.dump(data_train, train_filename)
        if not args.without_test:
            print("[SAVE]",test_filename)
            joblib.dump(data_test, test_filename)
        if not args.without_config:
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
            config_copy(args,config,config_fold,"plot_path",i)
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




if __name__ == '__main__':
    main()
