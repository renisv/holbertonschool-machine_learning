#!/usr/bin/env python3
"""
10-Adam.py
Creates a TensorFlow Adam optimizer
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow

    alpha: learning rate
    beta1: weight for the first moment
    beta2: weight for the second moment
    epsilon: small number to avoid division by zero
    Returns: TensorFlow optimizer
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
