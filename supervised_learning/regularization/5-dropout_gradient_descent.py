#!/usr/bin/env python3

import numpy as np

def dropout_gradient_descent(Y, weights, cache, L, alpha, keep_prob, lambtha, m):
    """
    Performs one pass of gradient descent with L2 regularization
    and dropout on the weights of a deep neural network.
    
    Arguments:
    Y -- one-hot labels, shape (classes, m)
    weights -- dictionary of weights and biases
    cache -- dictionary of outputs and dropout masks
    L -- number of layers
    alpha -- learning rate
    keep_prob -- probability of keeping a neuron
    lambtha -- L2 regularization parameter
    m -- number of examples

    Returns:
    weights -- updated weights dictionary
    """
    for l in reversed(range(1, L+1)):
        A_prev = cache['A' + str(l-1)]
        A = cache['A' + str(l)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        if l == L:
            dZ = A - Y
        else:
            dA = np.dot(weights['W' + str(l+1)].T, dZ) * cache['D' + str(l)]
            dA /= keep_prob
            dZ = dA * (A * (1 - A))

        dW = np.dot(dZ, A_prev.T)/m + (lambtha/m)*W
        db = np.sum(dZ, axis=1, keepdims=True)/m

        weights['W' + str(l)] = W - alpha*dW
        weights['b' + str(l)] = b - alpha*db

    return weights
