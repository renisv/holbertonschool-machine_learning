#!/usr/bin/env python3
"""Module that contains the function l2_reg_cost"""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost without L2 regularization
        model: Keras model that includes L2 regularization

    Returns:
        Tensor containing the total cost for each layer accounting
        for L2 regularization
    """
    reg_losses = model.losses
    return tf.stack([cost + loss for loss in reg_losses])
