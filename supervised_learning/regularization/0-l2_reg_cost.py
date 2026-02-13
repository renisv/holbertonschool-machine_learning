#!/usr/bin/env python3
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of weights and biases of the neural network
    L: number of layers
    m: number of data points

    Returns: cost accounting for L2 regularization
    """
    l2_sum = 0

    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        l2_sum += np.sum(np.square(W))

    l2_cost = (lambtha / (2 * m)) * l2_sum

    return cost + l2_cost
