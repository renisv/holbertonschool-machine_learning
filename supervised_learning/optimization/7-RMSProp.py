#!/usr/bin/env python3
"""
7-RMSProp.py
Updates variables using the RMSProp optimization algorithm
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp

    alpha: learning rate
    beta2: RMSProp weight
    epsilon: small number to avoid division by zero
    var: numpy.ndarray (or scalar) to be updated
    grad: numpy.ndarray (or scalar) gradient of var
    s: previous second moment
    Returns: updated variable, updated second moment
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
