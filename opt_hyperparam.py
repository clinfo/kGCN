from numpy.random import seed
import GPyOpt
import os
import multiprocessing
import numpy as np
import json
import argparse

opt_cmd="python3 gcn.py --config %s train"
domain = [
        {'name': 'num_gcn_layer', 'type': 'discrete', 'domain': (0,1,2)}
        ]
seed(123)

opt_config_path=None
opt_result_path=None
opt_param_path=None

# multiprocess is not supported
batch_size=1
num_cores=1

# global variable
counter=0

def save_json(path,obj):
    print("[SAVE] ",path)
    with open(path,"w") as fp:
        json.dump(obj,fp, indent=4)

def load_json(path):
    print("[LOAD] ",path)
    with open(path, 'r') as fp:
        obj=json.load(fp)
    return obj

def fx(x):
    #worker=multiprocessing.current_process()._identity
    global counter
    fid=counter
    counter+=1

    # build config
    opt_param_path=opt_path+"param."+str(fid)+".json"
    opt_result_path=opt_path+"result."+str(fid)+".json"
    config["save_info_valid"]=opt_result_path
    config["param"]=opt_param_path
    config["save_model"]=opt_path+"model."+str(fid)+".ckpt"
    opt_config_path=opt_path+"config."+str(fid)+".json"
    # save config
    save_json(opt_config_path,config)

    #  save parameters
    param={}
    for i,el in enumerate(domain):
        param[el["name"]]=x[0,i]
    save_json(opt_param_path,param)

    # exec command
    cmd=opt_cmd%(opt_config_path)
    print("cmd:",cmd)
    os.system(cmd)

    # get result
    result=load_json(opt_result_path)

    return result["validation_cost"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
            default=None,
            nargs='?',
            help='config json file')
    parser.add_argument('--max_itr', type=int,
            default=3,
            help='maximum iteration')
    parser.add_argument('--opt_path', type=str,
            default="opt/",
            help='path')
    parser.add_argument('--domain', type=str,
            default=None,
            help='domain file')
    args=parser.parse_args()
    # load config
    if args.config is None:
        parser.print_help()
        quit()
    else:
        config=load_json(args.config)

    print("... preparing optimization")
    # make directory
    opt_path=args.opt_path
    os.makedirs(opt_path,exist_ok=True)
    # load domain
    if args.domain is not None:
        domain=load_json(args.domain)
    print("... starting optimization")
    opt = GPyOpt.methods.BayesianOptimization(f=fx,
        domain = domain,
        batch_size = batch_size,
        num_cores = num_cores)

    opt.run_optimization(max_iter=args.max_itr)

    print("... saving optimized parameters")
    n=opt.X.shape[0]
    result=[]
    for i in range(n):
        param={}
        for j,el in enumerate(domain):
            param[el["name"]]=opt.X[i,j]
        y=opt.Y[i,0]
        result.append({"param":param,"cost":y})
    out_result_path=opt_path+"opt_result.json"
    save_json(out_result_path,result)
    opt_index=np.argmin(opt.Y[:,0])

    print("... saving optimized parameters")
    param={}
    for j,el in enumerate(domain):
        param[el["name"]]=opt.x_opt[j]
    print("optimized parapeter: ",param)
    print("cost : ",opt.fx_opt)
    print("index: ",opt_index)
    param["opt_index"]=int(opt_index)
    # save optimized parameters
    out_path=opt_path+"opt_param.json"
    save_json(out_path,param)

