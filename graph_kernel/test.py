# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
import joblib
import numpy as np

def main():
    # Load ENZYMES data set
  #  graph_db, classes = dp.read_txt("ENZYMES")
    obj=joblib.load("../sample.graph.jbl")
    graph_db=obj["graph"][:]
    classes=obj["label"][:,0]
    print(graph_db[0])
    print(classes)
    print(len(classes))
    print("*****")
    # Parameters used:
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False

    #kernel_parameters_sp = [False, False, 1]
    kernel_parameters_sp = [True, True, 1]

    g=sp_exp.shortest_path_kernel(graph_db, [],*kernel_parameters_sp)
    print(g.shape)
    np.save("gram_sp.npy",g)

if __name__ == "__main__":
    main()
