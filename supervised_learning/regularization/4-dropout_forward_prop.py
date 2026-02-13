#!/usr/bin/env python3

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Forward propagation with dropout.
    
    X: input data (nx, m)
    weights: dictionary of weights and biases
    L: number of layers
    keep_prob: probability of keeping a neuron active
    Returns: cache with activations and dropout masks
    """
    cache = {}
    A = X
    cache["A0"] = A  # <-- must be input data, not zeros
    
    for l in range(1, L + 1):
        W = weights['W{}'.format(l)]
        b = weights['b{}'.format(l)]
        Z = W @ A + b

        if l != L:
            A = np.tanh(Z)
            # create dropout mask
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob
            cache["D{}".format(l)] = D.astype(int)
        else:
            # softmax for last layer
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / np.sum(expZ, axis=0, keepdims=True)

        cache["A{}".format(l)] = A

    return cache
