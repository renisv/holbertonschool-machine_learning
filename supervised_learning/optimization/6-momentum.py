#!/usr/bin/env python3
"""
6-momentum.py
Creates a TensorFlow momentum optimizer
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimizer in TensorFlow

    alpha: learning rate
    beta1: momentum weight
    Returns: TensorFlow optimizer
    """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
