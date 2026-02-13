#!/usr/bin/env python3
"""5. Gradient descent with dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights: dictionary of weights and biases
    cache: dictionary of outputs and dropout masks
    alpha: learning rate
    keep_prob: probability a node is kept
    L: number of layers
    """
    m = Y.shape[1]
    dZ = {}

    # Last layer (softmax)
    A_L = cache['A' + str(L)]
    dZ[L] = A_L - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)] if l > 1 else cache['A0']
        dW = (1 / m) * np.dot(dZ[l], A_prev.T)
        db = (1 / m) * np.sum(dZ[l], axis=1, keepdims=True)

        # Update weights
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            # Backprop through dropout mask
            dA_prev = np.dot(weights['W' + str(l)].T, dZ[l])
            dA_prev *= cache['D' + str(l - 1)]  # Apply dropout mask
            dA_prev /= keep_prob
            # Backprop through tanh
            dZ[l - 1] = dA_prev * (1 - np.power(cache['A' + str(l - 1)], 2))
