# Author: Elisabetta Ghisu

"""

- This script take as input a kernel matrix
and returns the classification or regression performance

- The kernel matrix can be calculated using any of the graph kernels approaches

- The criteria used for prediction are SVM for classification and kernel Ridge regression for regression

- For predition we divide the data in training, validation and test. For each split, we first train on the train data, 
then evaluate the performance on the validation. We choose the optimal parameters for the validation set and finally
provide the corresponding perforance on the test set. If more than one split is performed, the final results 
correspond to the average of the performances on the test sets. 

"""

# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

###########################
# --- IMPORT PACKAGES --- #
###########################

import numpy as np
import pickle
import os
import argparse
import random

from numpy import genfromtxt

from sklearn.kernel_ridge import KernelRidge # 0.17
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import svm
from sklearn.model_selection import KFold
#from sklearn.grid_search import ParameterGrid
from sklearn.model_selection import ParameterGrid

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
from dataset2graph import dataset2graphset, concat_graphset
import joblib
import argparse
import numpy as np



def permutation_gram_matrix(K,y):
    n = K.shape[0]
    idx_perm = np.random.permutation(n)
    y_perm = y[idx_perm] #targets permutation
    K_perm = K[:,idx_perm] #inputs permutation
    K_perm = K_perm[idx_perm,:] #inputs permutation
    return K_perm,y_perm

def splice_gram_matrix(K,y,idx_train,idx_val,idx_test):
    #idx_perm = np.random.permutation(n)
    # Split the kernel matrix

    K_base = K[:,idx_train]
    K_train = K_base[idx_train,:]
    if idx_val is not None:
        K_val   = K_base[idx_val, :]
    else:
        K_val   = None

    if idx_test is not None:
        K_test  = K_base[idx_test, :]
    else:
        K_test  = None

    # Split the targets
    y_train = y[idx_train]
    if idx_val is not None:
        y_val = y[idx_val]
    else:
        y_val = None

    if idx_test is not None:
        y_test = y[idx_test]
    else:
        y_test  = None

    return K_train,y_train, K_val,y_val, K_test,y_test

def trial_fit(K,y,idx_train,idx_val,idx_test,grid):
    K_train,y_train, K_val,y_val, K_test,y_test=splice_gram_matrix(K,y,idx_train,idx_val,idx_test)
    print(len(idx_train),len(idx_val),len(idx_test))
    # For each parameter trial
    trials=len(grid)
    perf_all_val=[None for _ in range(trials)]
    perf_all_test=[None for _ in range(trials)]
    for i,param in enumerate(grid):
        # Fit classifier on training data
        clf = svm.SVC(kernel = 'precomputed', C = param["C"])
        clf.fit(K_train, y_train)

        # predict on validation and test
        y_pred = clf.predict(K_val)
        y_pred_test = clf.predict(K_test)

        # accuracy on validation set
        acc = accuracy_score(y_val, y_pred)
        perf_all_val[i]=acc

        # accuracy on test set
        acc_test = accuracy_score(y_test, y_pred_test)
        perf_all_test[i]=acc_test

        print("Trial= %d and C = %3f" % (i,param["C"]))
        print("Acc. (validation) : %3f" % acc)
        print("Acc. (test)       : %3f" % acc_test)
        print(y_test)
        print(y_pred_test)
    return perf_all_val,perf_all_test

def load_data(args):
    graph_db=None
    y=None
    idx_train=None
    idx_test=None
    limit_size=args.limit
    if args.input_train_dataset is not None:
        obj_train_dataset=joblib.load(args.input_train_dataset)
        obj_train=dataset2graphset(obj_train_dataset)
        obj_test_dataset=joblib.load(args.input_test_dataset)
        obj_test=dataset2graphset(obj_test_dataset)
        # checking train data size
        n=len(obj_train["graph"])
        idx_data=np.random.shuffle(list(range(n)))
        if n>limit_size:
            print("[WARN] too much training data: limited-%d"%(limit_size))
            n=limit_size
            idx_data=idx_data[:n]
        obj_train["graph"]=obj_train["graph"][idx_data]
        obj_train["label"]=obj_train["label"][idx_data,:]
        # concat training/test data
        obj,length_pair=concat_graphset(obj_train,obj_test)
        l1=length_pair[0]
        l2=length_pair[1]
        idx=list(range(l1+l2))
        idx_train=idx[:l1]
        idx_test=idx[l1:]
        graph_db=obj["graph"]
        y=obj["label"][:,0]
    else:
        if args.input_dataset is not None:
            obj_dataset=joblib.load(args.input_dataset)
            obj=dataset2graphset(obj_dataset)
        else:
            obj=joblib.load(args.input_graph)
        ### setup data
        data_num=len(obj["graph"])
        idx_data=np.array(list(range(data_num)))
        np.random.shuffle(idx_data)
        if data_num>limit_size:
            print("[WARN] too much data: limited-%d"%(limit_size))
            data_num=limit_size
            idx_data=idx_data[:data_num]
        graph_db=np.array(obj["graph"])
        graph_db=graph_db[idx_data]
        y=obj["label"][idx_data,0]
    return graph_db,y,idx_train,idx_test


def main():

    ##################################
    # --- COMMAND LINE ARGUMENTS --- #
    ##################################

    parser = argparse.ArgumentParser(description = "Classification/regression experiments with SP")
    parser.add_argument("--trials",default=5, help = "Trials for hyperparameters random search", type = int)
    parser.add_argument("--test_splits",default=5, help = "number of splits", type = int)
    parser.add_argument("--val_splits",default=5, help = "number of splits", type = int)

    parser.add_argument("--kernel",default="wl", help = "wl/sp/all", type = str)
    parser.add_argument("--input_graph",default="../sample.graph.jbl", help = "joblib file", type = str)
    parser.add_argument("--input_dataset",default=None, help = "joblib file", type = str)
    parser.add_argument("--input_train_dataset",default=None, help = "joblib file (train, validation)", type = str)
    parser.add_argument("--input_test_dataset",default=None, help = "joblib file (test)", type = str)
    parser.add_argument("--output",default="", help = "output path", type = str)
    parser.add_argument("--limit",default=3000, help = "limit data size", type = int)


    args = parser.parse_args()
    ### Load data set
    graph_db,y,idx_train_val,idx_test=load_data(args)

    ### computing kernel
    print("=================")
    wl_kernel_flag=False
    sp_kernel_flag=False
    if args.kernel=="wl":
        wl_kernel_flag=True
    elif args.kernel=="sp":
        sp_kernel_flag=True
    #elif args.kernel=="all":
    #	wl_kernel_flag=True
    #	sp_kernel_flag=True

    ###
    ### building gram matrix
    ###
    out_path=args.output
    K=None
    if sp_kernel_flag:
        # Parameters used:
        # Compute gram matrix: False,
        # Normalize gram matrix: False
        # Use discrete labels: False

        #kernel_parameters_sp = [False, False, 1]
        kernel_parameters_sp = [True, True, 1]
        K=sp_exp.shortest_path_kernel(graph_db, [],*kernel_parameters_sp)
    if wl_kernel_flag:
        #kernel_parameters_wl = [2,False, False, 1]
        kernel_parameters_wl = [3,True, True, 1]
        K=wl.weisfeiler_lehman_subtree_kernel(graph_db, [], *kernel_parameters_wl)

    ###
    ### SAVE internal data
    ###
    if out_path is not None:
        if sp_kernel_flag:
            print("[SAVE] "+out_path+"gram_sp.npy")
            np.save(out_path+"gram_sp.npy",K)
        if wl_kernel_flag:
            print("[SAVE] "+out_path+"gram_wl.npy")
            np.save(out_path+"gram_wl.npy",K)
        print("[SAVE] "+out_path+"label.npy")
        np.save(out_path+"label.npy",y)

    ###
    ### SVM
    ###
    random_state = 0
    np.random.seed(random_state)

    #################################
    # --- SET UP THE PARAMETERS --- #
    #################################

    C_grid = np.linspace(0.0001, 10, num = 5)
    param_grid = {'C': C_grid}
    grid = ParameterGrid(param_grid)

    if idx_train_val is None:
        # With the corresponding performance on test
        test_split = []
        data_num=K.shape[0]
        idx=list(range(data_num))
        # For each split of the data
        for idx_train_val,idx_test in KFold(n_splits=args.test_splits).split(idx):

            perf_all_val=[]
            perf_all_test=[]
            for idx_train,idx_val in KFold(n_splits=args.test_splits).split(idx_train_val):
                perf_val,perf_test=trial_fit(K,y,idx_train,idx_val,idx_test,grid)
                perf_all_val.append(perf_val)
                perf_all_test.append(perf_test)
            #######################################
            # --- FIND THE OPTIMAL PARAMETERS --- #
            #######################################
            perf_all_val = np.mean(perf_all_val,axis=0)
            perf_all_test = np.mean(perf_all_test,axis=0)
            max_idx = np.argmax(perf_all_val)
            param_opt = grid[max_idx]
            perf_val_opt = perf_all_val[max_idx]
            perf_test_opt = perf_all_test[max_idx]
            print("==============")
            print("[Best] Trial %d and C = %3f" % (max_idx,param_opt["C"]))
            print("[Best] Acc. (validation): %3f" % perf_val_opt)
            print("[Best] Acc. (test)      : %3f" % perf_test_opt)
            # append the correponding performance on the test set
            test_split.append(perf_test_opt)


        ###############################
        # --- AVERAGE THE RESULTS --- #
        ###############################
        test_mean = np.mean(np.asarray(test_split))
        test_std = np.std(np.asarray(test_split))
        print("=======================")
        print("Accuracy (test) mean: %3f" % test_mean)
        print("Accuracy (test) std.: %3f" % test_std)
    else:

        perf_all_val=[]
        perf_all_test=[]
        for idx_train,idx_val in KFold(n_splits=args.test_splits).split(idx_train_val):
            perf_val,perf_test=trial_fit(K,y,idx_train,idx_val,idx_test,grid)
            perf_all_val.append(perf_val)
            perf_all_test.append(perf_test)
        #######################################
        # --- FIND THE OPTIMAL PARAMETERS --- #
        #######################################
        perf_all_val = np.mean(perf_all_val,axis=0)
        perf_all_test = np.mean(perf_all_test,axis=0)
        max_idx = np.argmax(perf_all_val)
        param_opt = grid[max_idx]
        perf_val_opt = perf_all_val[max_idx]
        perf_test_opt = perf_all_test[max_idx]
        print("==============")
        print("[Best] Trial %d and C = %3f" % (max_idx,param_opt["C"]))
        print("[Best] Acc. (validation): %3f" % perf_val_opt)
        print("[Best] Acc. (test)      : %3f" % perf_test_opt)

if __name__ == "__main__":
    main()





