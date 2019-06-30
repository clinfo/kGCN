# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import graph_tool.all as gt
#import graph_tool as gt
import numpy as np
import scipy as sp
import scipy.sparse.csr as csr
import scipy.sparse.lil as lil

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import log_primes_list as log_pl


def weisfeiler_lehman_subtree_kernel(graph_db, hashed_attributes, *kwargs):
    iterations = kwargs[0]
    compute_gram_matrix = kwargs[1]
    normalize_gram_matrix = kwargs[2]
    use_labels = kwargs[3]

    # Create one empty feature vector for each graph
    feature_vectors = []
    for _ in graph_db:
        feature_vectors.append(np.zeros(0, dtype=np.float64))

    # Construct block diagonal matrix of all adjacency matrices
    adjacency_matrices = []
    for g in graph_db:
        adjacency_matrices.append(gt.adjacency(g))
    M = sp.sparse.block_diag(tuple(adjacency_matrices), dtype=np.float64, format="csr")
    num_vertices = M.shape[0]

    # Load list of precalculated logarithms of prime numbers
    log_primes = log_pl.log_primes[0:num_vertices]

    # Color vector representing labels
    colors_0 = np.zeros(num_vertices, dtype=np.float64)
    # Color vector representing hashed attributes
    colors_1 = hashed_attributes

    # Get labels (colors) from all graph instances
    offset = 0
    graph_indices = []

    for g in graph_db:
        if use_labels == 1:
            for i, v in enumerate(g.vertices()):
                colors_0[i + offset] = g.vp.nl[v]
        if use_labels == 2:
            for i, v in enumerate(g.vertices()):
                colors_0[i + offset] = v.out_degree()

        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    # Map labels to [0, number_of_colors)
    if use_labels:
        _, colors_0 = np.unique(colors_0, return_inverse=True)

    for it in range(0, iterations + 1):

        if use_labels:
            # Map colors into a single color vector
            if len(colors_1)>0:
                colors_all = np.array([colors_0, colors_1])
            else:
                colors_all = np.array([colors_0])
            colors_all = [hash(tuple(row)) for row in colors_all.T]
            _, colors_all = np.unique(colors_all, return_inverse=True)
            max_all = int(np.amax(colors_all) + 1)
            # max_all = int(np.amax(colors_0) + 1)

            feature_vectors = [
                np.concatenate((feature_vectors[i], np.bincount(colors_all[index[0]:index[1] + 1], minlength=max_all)))
                for i, index in enumerate(graph_indices)]

            # Avoid coloring computation in last iteration
            if it < iterations:
                colors_0 = compute_coloring(M, colors_0, log_primes[0:len(colors_0)])
                if len(colors_1)>0:
                    colors_1 = compute_coloring(M, colors_1, log_primes[0:len(colors_1)])
        else:
            max_1 = int(np.amax(colors_1) + 1)

            feature_vectors = [
                np.concatenate((feature_vectors[i], np.bincount(colors_1[index[0]:index[1] + 1], minlength=max_1))) for
                i, index in enumerate(graph_indices)]

            # Avoid coloring computation in last iteration
            if it < iterations:
                colors_1 = compute_coloring(M, colors_1, log_primes[0:len(colors_1)])

    if not compute_gram_matrix:
        return feature_vectors
	#return lil.lil_matrix(feature_vectors, dtype=np.float64)
    else:
        # Make feature vectors sparse
        gram_matrix = csr.csr_matrix(feature_vectors, dtype=np.float64)
        # Compute gram matrix
        gram_matrix = gram_matrix.dot(gram_matrix.T)

        gram_matrix = gram_matrix.toarray()

        if normalize_gram_matrix:
            return aux.normalize_gram_matrix(gram_matrix)
        else:
            return gram_matrix


def compute_coloring(M, colors, log_primes):
    log_prime_colors = np.array([log_primes[i] for i in colors], dtype=np.float64)
    colors = colors + M.dot(log_prime_colors)

    # Round numbers to avoid numerical problems
    colors = np.round(colors, decimals=10)

    _, colors = np.unique(colors, return_inverse=True)

    return colors
