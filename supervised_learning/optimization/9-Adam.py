#!/usr/bin/env python3
"""
9-Adam.py
Updates variables using the Adam optimization algorithm
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm

    alpha: learning rate
    beta1: weight for the first moment
    beta2: weight for the second moment
    epsilon: small number to avoid division by zero
    var: numpy.ndarray (or scalar) variable to update
    grad: numpy.ndarray (or scalar) gradient of var
    v: previous first moment
    s: previous second moment
    t: time step for bias correction (starts at 1)
    Returns: updated variable, updated first moment, updated second moment
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
