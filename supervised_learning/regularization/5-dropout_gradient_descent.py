#!/usr/bin/env python3
"""
Module for performing gradient descent on a neural network
with dropout regularization.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights and biases of a neural network
    with Dropout regularization using gradient descent.

    Y: one-hot numpy.ndarray of shape (classes, m)
    weights: dictionary of weights and biases
    cache: dictionary of activations and dropout masks
    alpha: learning rate
    keep_prob: probability that a neuron is kept
    L: number of layers
    """
    m = Y.shape[1]
    dA = cache["A{}".format(L)] - Y  # derivative of softmax cross-entropy

    for layer in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(layer - 1)]
        W = weights["W{}".format(layer)]

        dZ = dA
        if layer != L:
            D = cache["D{}".format(layer)]
            dZ = dA * (1 - cache["A{}".format(layer)] ** 2)  # tanh derivative
            dZ = dZ * D
            dZ = dZ / keep_prob

        dW = dZ @ A_prev.T / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db

        dA = W.T @ dZ
