#!/usr/bin/env python3
"""Module that contains the function l2_reg_gradient_descent"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m)
        weights: dictionary of weights and biases
        cache: dictionary of outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers in the network

    The weights and biases are updated in place
    """
    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            A_prev_layer = cache["A{}".format(i - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev_layer ** 2)

        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db
