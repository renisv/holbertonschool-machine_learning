#!/usr/bin/env python3
"""
1-sensitivity.py
Calculates the sensitivity (recall) for each class
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class

    confusion: numpy.ndarray of shape (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
    """
    true_positives = np.diag(confusion)
    possible_positives = np.sum(confusion, axis=1)

    return true_positives / possible_positives
