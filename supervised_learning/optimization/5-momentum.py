#!/usr/bin/env python3
"""
5-momentum.py
Updates variables using gradient descent with momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum algorithm

    alpha: learning rate
    beta1: momentum weight
    var: numpy.ndarray (or scalar) containing the variable to update
    grad: numpy.ndarray (or scalar) containing the gradient of var
    v: previous first moment (momentum term)
    Returns: updated variable, updated moment
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v

    return var, v
