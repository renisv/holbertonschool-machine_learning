#!/usr/bin/env python3
"""
8-RMSProp.py
Creates a TensorFlow RMSprop optimizer
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimizer in TensorFlow

    alpha: learning rate
    beta2: RMSProp weight (discounting factor)
    epsilon: small number to avoid division by zero
    Returns: TensorFlow optimizer
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
