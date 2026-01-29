#!/usr/bin/env python3
"""
25-one_hot_decode.py

Defines a function that converts a one-hot encoded matrix into labels.
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot encoded matrix into a vector of labels.

    Args:
        one_hot (np.ndarray): One-hot matrix of shape (classes, m).

    Returns:
        np.ndarray or None: Array of shape (m,) with numeric labels,
        or None on failure.
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.ndim != 2:
        return None

    classes, m = one_hot.shape

    if classes == 0 or m == 0:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    if not np.all(np.sum(one_hot, axis=0) == 1):
        return None

    return np.argmax(one_hot, axis=0)
