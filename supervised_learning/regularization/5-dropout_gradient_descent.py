#!/usr/bin/env python3
"""
Gradient descent with dropout
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights and biases of a neural network using dropout gradient descent.
    """
    m = Y.shape[1]
    dA = cache["A{}".format(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(l - 1)]
        W = weights["W{}".format(l)]

        dZ = dA
        if l != L:
            D = cache["D{}".format(l)]
            dZ = dA * (1 - cache["A{}".format(l)] ** 2)  # tanh derivative
            dZ = dZ * D
            dZ = dZ / keep_prob

        dW = dZ @ A_prev.T / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(l)] -= alpha * dW
        weights["b{}".format(l)] -= alpha * db

        dA = W.T @ dZ
