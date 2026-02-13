#!/usr/bin/env python3

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with Dropout.

    X: input data, shape (nx, m)
    weights: dictionary of weights and biases
    L: number of layers
    keep_prob: probability of keeping a node
    Returns: cache dictionary with activations and dropout masks
    """
    cache = {}
    A = X
    cache["A0"] = np.zeros_like(X)  # A0 should be zeros
    for l in range(1, L + 1):
        W = weights['W{}'.format(l)]
        b = weights['b{}'.format(l)]
        Z = W @ A + b

        if l != L:
            A = np.tanh(Z)
            # Dropout mask
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob
            cache["D{}".format(l)] = D.astype(int)
        else:
            # last layer uses softmax
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / expZ.sum(axis=0, keepdims=True)
        cache["A{}".format(l)] = A
    return cache
