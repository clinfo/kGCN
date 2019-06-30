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


##################################
# --- COMMAND LINE ARGUMENTS --- #
##################################

parser = argparse.ArgumentParser(description = "Classification/regression experiments with SP")
parser.add_argument("--trials",default=5, help = "Trials for hyperparameters random search", type = int)
parser.add_argument("--splits",default=5, help = "number of splits", type = int)


args = parser.parse_args()

# Number of parameter trials
trials = "%d" % (args.trials) 
trials = int(trials)

# Set the seed for uniform parameter distribution
random.seed(20) 


# Number of splits of the data
splits = args.splits


#########################
# --- LOAD THE DATA --- #
######################### 

# Load the kernel matrix
filename="gram_wl.npy"
print("Loading kernel matrix...")
K = np.load(filename)

# Path to the targets
label_name = "label.npy"
# Load targets
y = np.load(label_name)

# Size of the dataset
n = K.shape[0]


#################################
# --- SET UP THE PARAMETERS --- #
#################################

# You should modify these arguments
# depending on the range of parameters that you want to examine
alpha_grid = np.linspace(0.01, 100, num = trials)
C_grid = np.linspace(0.0001, 10, num = trials)


##############################################################
# --- MAIN CODE: PERMUTE, SPLIT AND EVALUATE PEFORMANCES --- #
##############################################################


"""

-  Here starts the main program

-  First we permute the data, then
for each split we evaluate corresponding performances

-  In the end, the performances are averaged over the test sets

"""

# Initialize the performance of the best parameter trial on validation
# With the corresponding performance on test
val_split = []
test_split = []

# For each split of the data
for j in range(10, 10 + splits):

    print("==========================")
    print("Starting split %d..." % j)

    # Set the random set for data permutation
    random_state = int(j)
    np.random.seed(random_state)
    idx_perm = np.random.permutation(n)

    # Set the output path
    output_path = "output"

    # if the output directory does not exist, then create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # Permute the data
    y_perm = y[idx_perm] #targets permutation
    K_perm = K[:,idx_perm] #inputs permutation
    K_perm = K_perm[idx_perm,:] #inputs permutation

    # Set the training, validation and test
    # Note: the percentage can be set up by the user
    num_train_val = int((n * 90)/100)         #90% (of entire dataset) for training and validation
    num_test = n - num_train_val              #10% (of entire dataset) for test
    num_train = int((num_train_val * 90)/100) #90% (of train + val) for training
    num_val = num_train_val - num_train       # ~10% (of train + val) for validation

    # Split the kernel matrix
    K_train = K_perm[0:num_train,0:num_train]
    K_val = K_perm[num_train:(num_train+num_val),0:num_train]
    K_test = K_perm[(num_train+num_val):n, 0:num_train]

    # Split the targets
    y_train = y_perm[0:num_train]

    y_val = y_perm[num_train:(num_train+num_val)]
    y_test = y_perm[(num_train+num_val):n]

    # Record the performance for each parameter trial
    # respectively on validation and test set
    perf_all_val = []
    perf_all_test = []


    #####################################################################
    # --- RUN THE MODEL: FOR A GIVEN SPLIT AND EACH PARAMETER TRIAL --- #
    #####################################################################

    # For each parameter trial
    for i in range(trials):


            # Fit classifier on training data
            clf = svm.SVC(kernel = 'precomputed', C = C_grid[i])
            clf.fit(K_train, y_train)

            # predict on validation and test
            y_pred = clf.predict(K_val)
            y_pred_test = clf.predict(K_test)

            # accuracy on validation set
            acc = accuracy_score(y_val, y_pred)
            perf_all_val.append(acc)

            # accuracy on test set
            acc_test = accuracy_score(y_test, y_pred_test)
            perf_all_test.append(acc_test)

            print("Trial= %d and C = %3f" % (i, C_grid[i]))
            print("The acc. on the validation set is: %3f" % acc)
            print("The acc. on the test set is: %3f" % acc_test)


    #######################################
    # --- FIND THE OPTIMAL PARAMETERS --- #
    #######################################

    # For classification: maximise the accuracy
    # get optimal parameter on validation (argmax accuracy)
    max_idx = np.argmax(perf_all_val)
    C_opt = C_grid[max_idx]

    # performance corresponsing to the optimal parameter on validation
    perf_val_opt = perf_all_val[max_idx]

    # corresponding performance on the test set for the same parameter
    perf_test_opt = perf_all_test[max_idx]

    print("==============")
    print("[Best] Trial %d and C = %3f" % (max_idx, C_opt))
    print("[Best] The acc. on the validation set is: %3f" % perf_val_opt)
    print("[Best] The acc. on test set is: %3f" % perf_test_opt)


    #######################
    # --- SAVE RESULTS ---#
    #######################

    # file to save performances
    f_perf = "%s/kernel_perf_%d.txt" % (output_path,j)
    f_out = open(f_perf,'w')

    # Write on file: random state, iteration of WL,
    # performances on validation and test set for each parameter trial
    f_out.write("Random state = %d\n" % random_state)

    #if graph_kernel == "wl":
    #	f_out.write("h of WL = %d\n" % h)
    f_out.write("Performance on the validation and test set for different parameter trials \n \n")
    f_out.write("trial \t param \t \t perf val \t perf test\n")
    for j in range(trials):
        f_out.write("%d \t %3f \t %3f \t %3f \n" % (j, C_grid[j], perf_all_val[j], perf_all_test[j]))
    f_out.close()

    # append the best performance on validation
    # at the current split
    val_split.append(perf_val_opt)

    # append the correponding performance on the test set
    test_split.append(perf_test_opt)


###############################
# --- AVERAGE THE RESULTS --- #
###############################

# mean of the validation performances over the splits
val_mean = np.mean(np.asarray(val_split))
# std deviation of validation over the splits
val_std = np.std(np.asarray(val_split))

# mean of the test performances over the splits
test_mean = np.mean(np.asarray(test_split))
# std deviation of the test oer the splits
test_std = np.std(np.asarray(test_split))

print("=======================")
print("\n Mean performance on val set: %3f" % val_mean)
print("With standard deviation: %3f" % val_std)
print("\n Mean performance on test set: %3f" % test_mean)
print("With standard deviation: %3f" % test_std)







