#!/usr/bin/env python3
"""
24-one_hot_encode.py

Defines a function that converts numeric labels to a one-hot matrix.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot encoded matrix.

    Args:
        Y (np.ndarray): Array of shape (m,) with numeric class labels.
        classes (int): Total number of classes.

    Returns:
        np.ndarray or None: One-hot encoded matrix of shape (classes, m),
        or None on failure.
    """
    if not isinstance(Y, np.ndarray):
        return None
    if Y.ndim != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None
    if np.any(Y < 0) or np.any(Y >= classes):
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))

    for i in range(m):
        one_hot[Y[i], i] = 1

    return one_hot
