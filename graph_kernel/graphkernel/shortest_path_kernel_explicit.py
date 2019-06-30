# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import graph_tool.all as gt
#import graph_tool as gt
import itertools as it
import numpy as np
import scipy.sparse.csgraph as csg
import scipy.sparse.csr as csr
import scipy.sparse.lil as lil

from auxiliarymethods import auxiliary_methods as aux


def shortest_path_kernel(graph_db, hashed_attributes, *kwargs):
    compute_gram_matrix = kwargs[0]
    normalize_gram_matrix = kwargs[1]
    use_labels = kwargs[2]

    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()

    offset = 0
    graph_indices = []
    colors_0 = np.zeros(num_vertices, dtype=np.int64)

    # Get labels (colors) from all graph instances
    offset = 0
    for g in graph_db:
        graph_indices.append((offset, offset + g.num_vertices() - 1))

        if use_labels == 1:
            for i, v in enumerate(g.vertices()):
                colors_0[i + offset] = g.vp.nl[v]
        if use_labels == 2:
            for i, v in enumerate(g.vertices()):
                colors_0[i + offset] = v.out_degree()

        offset += g.num_vertices()
    _, colors_0 = np.unique(colors_0, return_inverse=True)

    colors_1 = hashed_attributes

    triple_indices = []
    triple_offset = 0
    triples = []

    # Solve APSP problem for every graphs in graph data base
    for i, g in enumerate(graph_db):
        a = gt.adjacency(g)
        M = csg.shortest_path(a, method='J', directed=False, unweighted=True)

        index = graph_indices[i]

        if use_labels:
            l = colors_0[index[0]:index[1] + 1]
            h = colors_1[index[0]:index[1] + 1]
        else:
            h = colors_1[index[0]:index[1] + 1]
        d = M.shape[0]

        # For each pair of vertices collect labels, hashed attributes, and shortest-path distance
        pairs = list(it.product(list(range(d)), repeat=2))
        if use_labels:
			#t = [hash((l[k], h[k], l[j], h[j], M[k][j])) for (k, j) in pairs if (k != j or ~np.isinf(M[k][j]))]
            t = [hash((l[k], l[j], M[k][j])) for (k, j) in pairs if (k != j or ~np.isinf(M[k][j]))]
			#t = [ M[k][j] for (k, j) in pairs if (k != j or ~np.isinf(M[k][j]))]
        else:
            t = [hash((h[k], h[j], M[k][j])) for (k, j) in pairs if (k != j or ~np.isinf(M[k][j]))]

        triples.extend(t)

        triple_indices.append((triple_offset, triple_offset + len(t) - 1))
        triple_offset += len(t)

    _, colors = np.unique(triples, return_inverse=True)
    m = np.amax(colors) + 1

    # Compute feature vectors
    feature_vectors = []
    for i, index in enumerate(triple_indices):
        feature_vectors.append(np.bincount(colors[index[0]:index[1] + 1], minlength=m))

    if not compute_gram_matrix:
        return lil.lil_matrix(feature_vectors, dtype=np.float64)
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
