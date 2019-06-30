# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
import joblib

def main():
    # Load ENZYMES data set
  #  graph_db, classes = dp.read_txt("ENZYMES")
    obj=joblib.load("../sample.graph.jbl")
    graph_db=obj["graph"]
    classes=obj["label"]
    print(graph_db[0])
    print(classes)
    print(len(classes))
    print("*****1")
    # Parameters used:
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_sp = [False, False, 1]

    # Parameters used:
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False
    # Number of iterations for WL: 3
    kernel_parameters_wl = [3, False, False, 1]

    # Use discrete labels, too
    # kernel_parameters_sp = [False, False, 1]
    # kernel_parameters_wl = [3, False, False, 1]


    # Compute gram matrix for HGK-WL
    # 20 is the number of iterations
    gram_matrix = rbk.hash_graph_kernel(graph_db, sp_exp.shortest_path_kernel, kernel_parameters_sp, 20,
                                        scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)
    # Normalize gram matrix
    gram_matrix = aux.normalize_gram_matrix(gram_matrix)
    print("****1")

    # Compute gram matrix for HGK-SP
    # 20 is the number of iterations
    gram_matrix = rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel, kernel_parameters_wl, 20,
                                        scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)

    # Normalize gram matrix
    gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    # Write out LIBSVM matrix
    dp.write_lib_svm(gram_matrix, classes, "gram_matrix")


if __name__ == "__main__":
    main()
