# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np


def normalize_gram_matrix(gram_matrix):
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.float64)

    for i in range(0, n):
        for j in range(i, n):
            if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g

    return gram_matrix_norm


def locally_sensitive_hashing(m, d, w, sigma=1.0):
    # Compute random projection vector
    v = np.random.randn(d, 1) * sigma  # / np.random.randn(d, 1)

    # Compute random offset
    b = w * np.random.rand() * sigma

    # Compute hashes
    labels = np.floor((np.dot(m, v) + b) / w)

    # Compute label
    _, indices = np.unique(labels, return_inverse=True)

    return indices
