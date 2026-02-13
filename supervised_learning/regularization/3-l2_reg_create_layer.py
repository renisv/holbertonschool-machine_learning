#!/usr/bin/env python3
"""Module that contains the function l2_reg_create_layer"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer with L2 regularization

    Args:
        prev: output tensor of previous layer
        n: number of nodes in the layer
        activation: activation function
        lambtha: L2 regularization parameter

    Returns:
        Output of the new layer
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )
    return layer(prev)
