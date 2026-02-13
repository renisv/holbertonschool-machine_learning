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
    cache["A0"] = A  # input data

    for layer in range(1, L + 1):
        W = weights['W{}'.format(layer)]
        b = weights['b{}'.format(layer)]
        Z = W @ A + b

        if layer != L:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob
            cache["D{}".format(layer)] = D.astype(int)
        else:
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / np.sum(expZ, axis=0, keepdims=True)

        cache["A{}".format(layer)] = A

    return cache
