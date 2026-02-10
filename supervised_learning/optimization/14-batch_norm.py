#!/usr/bin/env python3
"""
14-batch_norm.py
Creates a batch normalization layer in TensorFlow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network

    prev: activated output of the previous layer
    n: number of nodes in the layer to create
    activation: activation function to use on the output
    Returns: tensor of the activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    bn = tf.keras.layers.BatchNormalization(
        axis=1,
        momentum=0.9,
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(dense)

    return activation(bn)
