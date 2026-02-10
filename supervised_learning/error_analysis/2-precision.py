#!/usr/bin/env python3
"""
2-precision.py
Calculates the precision for each class
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class

    confusion: numpy.ndarray of shape (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
