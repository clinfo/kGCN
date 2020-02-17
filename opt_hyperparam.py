from numpy.random import seed
import GPyOpt
import os
import string
import multiprocessing
import numpy as np
import json
import argparse

# Global variables
opt_cmd = string.Template("kgcn --config ${config} train ${args}")
domain = [
    {'name': 'num_gcn_layer', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4), "data_type": "int"},
    {'name': 'layer_dim0', 'type': 'continuous', 'domain': (0.5, 3)},
    {'name': 'layer_dim1', 'type': 'continuous', 'domain': (0.5, 3)},
    {'name': 'layer_dim2', 'type': 'continuous', 'domain': (0.5, 3)},
    {'name': 'layer_dim3', 'type': 'continuous', 'domain': (0.5, 3)},
    {'name': 'add_dense0', 'type': 'discrete', 'domain': (0, 1), "data_type": "int"},
    {'name': 'add_dense1', 'type': 'discrete', 'domain': (0, 1), "data_type": "int"},
    {'name': 'add_dense2', 'type': 'discrete', 'domain': (0, 1), "data_type": "int"},
    {'name': 'add_dense3', 'type': 'discrete', 'domain': (0, 1), "data_type": "int"},
    {'name': 'num_dense_layer', 'type': 'discrete',  'domain': (0, 1, 2), "data_type": "int"},
    {'name': 'layer_dense_dim0', 'type': 'continuous', 'domain': (0.5, 3)},
    {'name': 'layer_dense_dim1', 'type': 'continuous', 'domain': (0.5, 3)},
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 0.001)},
    {'name': 'batch_size',    'type': 'discrete',   'domain': (10, 50, 100), "data_type": "int"},
    {'name': 'dropout_rate',  'type': 'continuous', 'domain': (0, 0.9)},
    ]
seed(123)
opt_path = None
opt_arg = ""
config = None
counter = 0
# multiprocess is not supported
batch_size = 1
num_cores = 1
#


def save_json(path, obj):
    print("[SAVE] ", path)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4)


def load_json(path):
    print("[LOAD] ", path)
    with open(path, 'r') as fp:
        obj = json.load(fp)
    return obj


def update_config(path, config, fid, key):
    if key in config:
        config[key] = os.path.join(path, os.path.basename(config[key]))


def make_config(path, config, fid):
    config["param"] = os.path.join(path, "param.json")
    config["save_info_valid"] = os.path.join(path, "result.json")
    config["save_model"] = os.path.join(path, f"model.{str(fid)}.ckpt")
    ###
    config["plot_path"] = path
    update_config(path, config, fid, "save_info_train")
    update_config(path, config, fid, "save_info_test")
    update_config(path, config, fid, "save_result_train")
    update_config(path, config, fid, "save_result_test")
    update_config(path, config, fid, "save_result_valid")
    ###
    return config


def fx(x):
    # worker=multiprocessing.current_process()._identity
    global counter
    global config
    fid = counter
    counter += 1

    # build config
    config = config
    opt_config_path = os.path.join(opt_path, f"config.{str(fid)}.json")
    path = os.path.join(opt_path, f"trial{fid:03d}")
    opt_result_path = os.path.join(path, "result.json")
    os.makedirs(path, exist_ok=True)
    # save config and  save parameters
    config = make_config(path, config, fid)
    param = {}
    for i, el in enumerate(domain):
        param[el["name"]] = x[0, i]
        if el["name"] in config:
            print(el["name"], "<=", x[0, i])
            if "data_type" in el and el["data_type"] == "int":
                config[el["name"]] = int(x[0, i])
            else:
                config[el["name"]] = x[0, i]
    save_json(opt_config_path, config)
    save_json(config["param"], param)

    # exec command
    context = {"config": opt_config_path, "args": opt_arg}
    cmd = opt_cmd.substitute(context)
    print("cmd:", cmd)
    os.system(cmd)

    # get result
    result = load_json(opt_result_path)

    return result["validation_cost"]


def main():
    global opt_path
    global config
    global opt_arg
    global domain
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, nargs='?', required=True,
                        help='config json file')
    parser.add_argument('--max_itr', type=int, default=3,
                        help='maximum iteration')
    parser.add_argument('--opt_path', type=str, default="opt/",
                        help='path')
    parser.add_argument('--domain', type=str, default=None,
                        help='domain file')
    parser.add_argument('--gpu', type=str, default=None,
                        help='[kgcn arg]')
    parser.add_argument('--cpu', action='store_true',
                        help='[kgcn arg]')
    args = parser.parse_args()

    config = load_json(args.config)

    opt_arg += f" --gpu {args.gpu}" if args.gpu else ""
    opt_arg += f" --cpu" if args.cpu else ""

    print("... preparing optimization")
    # make directory
    opt_path = args.opt_path
    os.makedirs(opt_path, exist_ok=True)
    # load domain
    if args.domain is not None:
        domain = load_json(args.domain)
    print("... starting optimization")
    opt = GPyOpt.methods.BayesianOptimization(f=fx,
                                              domain=domain,
                                              batch_size=batch_size,
                                              num_cores=num_cores)
    opt.run_optimization(max_iter=args.max_itr)

    print("... saving optimization result")
    n = opt.X.shape[0]
    result = []
    for i in range(n):
        param = {}
        for j, el in enumerate(domain):
            param[el["name"]] = opt.X[i, j]
        y = opt.Y[i, 0]
        result.append({"param": param, "cost": y})
    out_result_path = os.path.join(opt_path, "opt_result.json")
    save_json(out_result_path, result)
    opt_index = np.argmin(opt.Y[:, 0])

    # save optimized parameters
    print("... saving optimized parameters")
    param = {}
    for j, el in enumerate(domain):
        param[el["name"]] = opt.x_opt[j]
    print(f"optimized parapeter: {param}"
          f"cost: {opt.fx_opt}"
          f"index: {opt_index}")
    param["opt_index"] = int(opt_index)
    out_path = os.path.join(opt_path, "opt_param.json")
    save_json(out_path, param)

    # save optimized config
    print("... saving config")
    fid=int(opt_index)
    path = os.path.join(opt_path, f"trial{fid:03d}")
    #opt_config = make_config(path, config, fid)
    #config["save_model"] = os.path.join(path, f"model.{str(fid)}.ckpt")
    config["load_model"] = os.path.join(path, f"model.{str(fid)}.ckpt")
    opt_config_path = os.path.join(opt_path, f"opt_config.json")
    save_json(out_config_path, opt_config)

if __name__ == '__main__':
    main()
