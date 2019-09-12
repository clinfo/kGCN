import joblib
import sys
import json
import re
import argparse
import numpy as np
import copy
import os

from gcn import get_default_config

def config_copy(src,dest,key,v):
    data=src[key]
    path, ext = os.path.splitext(data)
    dest[key]=path+"."+v+ext

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
    parser.add_argument('--dataset', type=str,
            default=None,
            nargs='+',
            help='dataset')

    args=parser.parse_args()
    # config
    config=get_default_config()
    if args.config is None:
        pass
        parser.print_help()
        quit()
    else:
        print("[LOAD] ",args.config)
        fp = open(args.config, 'r')
        config.update(json.load(fp))
    # option
    for dataset in args.dataset:
        dname, _ = os.path.splitext( os.path.basename(dataset) )
        dconfig=copy.deepcopy(config)
        print("[INFO]",dataset)
        dconfig["dataset"]=dataset
        config_copy(config,dconfig,"save_model",dname)
        config_copy(config,dconfig,"load_model",dname)
        config_copy(config,dconfig,"save_result_test",dname)
        config_copy(config,dconfig,"save_result_train",dname)
        name, ext = os.path.splitext( args.config )
        config_filename=name+"."+dname+ext
        print("[SAVE]",config_filename)
        fp=open(config_filename,"w")
        json.dump(dconfig,fp, indent=4)

    quit()

