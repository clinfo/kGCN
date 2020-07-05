import joblib
import sys
import json
import re
import argparse
import numpy as np
import copy
import os
import kgcn.data_util

def main():
    parser = argparse.ArgumentParser()
    """
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
    """
    parser.add_argument('--input', type=str,
            default=None,
            nargs='+',
            help='dataset')
    parser.add_argument('--output', type=str,
            default="out.jbl",
            help='dataset')
    parser.add_argument('--fold', type=int,
            default=5,
            help='#fold')
    parser.add_argument('--seed', type=int,
            default=1234,
            help='seed')
    args=parser.parse_args()

    obj_list=[]
    for dataset_name in args.input:
        print("[LOAD]",dataset_name)
        obj=joblib.load(dataset_name)
        print("input keys:",obj.keys())
        data_num=kgcn.data_util.get_data_num_jbl_obj(obj)
        print("#data:",data_num)
        obj_list.append(obj)
    ###
    data=kgcn.data_util.join_jbl_obj(obj_list[0],obj_list[1])
    for k,v in data.items():
        print(k,len(v))
    """
    for train_idx, test_idx in splitter:
        ## setting dataset
        name, ext = os.path.splitext( os.path.basename(dataset_name) )
        train_filename=cv+"/"+name+".train_"+str(i)+".jbl"
        test_filename=cv+"/"+name+".test_"+str(i)+".jbl"
        if not args.without_train:
            print("[SAVE]",train_filename)
            joblib.dump(data_train, train_filename)
        if not args.without_test:
            print("[SAVE]",test_filename)
            joblib.dump(data_test, test_filename)
    """

if __name__ == '__main__':
    main()
