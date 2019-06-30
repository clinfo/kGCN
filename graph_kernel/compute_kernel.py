# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
from dataset2graph import dataset2graphset
import joblib
import argparse
import numpy as np



parser = argparse.ArgumentParser(description = "graph kernek")
parser.add_argument("--kernel",default="all", help = "wl/sp/all", type = str)
parser.add_argument("--input_graph",default="../sample.graph.jbl", help = "joblib file", type = str)
parser.add_argument("--input_dataset",default=None, help = "joblib file", type = str)
parser.add_argument("--output",default="", help = "output path", type = str)
parser.add_argument("--limit",default=3000, help = "limit data size", type = int)


def main():
    args = parser.parse_args()
    # Load ENZYMES data set
    if args.input_dataset is None:
        obj=joblib.load(args.input_graph)
    else:
        obj_dataset=joblib.load(args.input_dataset)
        obj=dataset2graphset(obj_dataset)
        print(obj.keys())
    limit_size=args.limit
    data_num=len(obj["graph"])
    if data_num>limit_size:
        print("[WARN] too much data: limited-%d"%(limit_size))
        data_num=limit_size
    graph_db=obj["graph"][:data_num]
    classes=obj["label"][:data_num,0]
    print("*****")
    wl_kernel_flag=False
    sp_kernel_flag=False
    if args.kernel=="wl":
        wl_kernel_flag=True
    elif args.kernel=="sp":
        sp_kernel_flag=True
    elif args.kernel=="all":
        wl_kernel_flag=True
        sp_kernel_flag=True

    out_path=args.output
    if sp_kernel_flag:
        # Parameters used:
        # Compute gram matrix: False,
        # Normalize gram matrix: False
        # Use discrete labels: False

        #kernel_parameters_sp = [False, False, 1]
        kernel_parameters_sp = [True, True, 1]
        g=sp_exp.shortest_path_kernel(graph_db, [],*kernel_parameters_sp)
        print("[SAVE] "+out_path+"gram_sp.npy")
        np.save(out_path+"gram_sp.npy",g)
    if wl_kernel_flag:
        #kernel_parameters_wl = [2,False, False, 1]
        kernel_parameters_wl = [3,True, True, 1]
        g=wl.weisfeiler_lehman_subtree_kernel(graph_db, [], *kernel_parameters_wl)
        print("[SAVE] "+out_path+"gram_wl.npy")
        np.save(out_path+"gram_wl.npy",g)
    #joblib.dump(g,"gram.jbl")
    print("[SAVE] "+out_path+"label.npy")
    np.save(out_path+"label.npy",classes)

if __name__ == "__main__":
    main()
