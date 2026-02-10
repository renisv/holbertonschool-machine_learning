#!/usr/bin/env python3
"""
11-learning_rate_decay.py
Updates the learning rate using inverse time decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay (stepwise)

    alpha: original learning rate
    decay_rate: decay rate
    global_step: number of gradient descent steps elapsed
    decay_step: number of steps before further decay
    Returns: updated learning rate
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
