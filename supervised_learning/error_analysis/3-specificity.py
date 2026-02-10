#!/usr/bin/env python3
"""
3-specificity.py
Calculates the specificity for each class
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class

    confusion: numpy.ndarray of shape (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
    """
    total = np.sum(confusion)
    true_positives = np.diag(confusion)

    false_positives = (
        np.sum(confusion, axis=0) - true_positives
    )
    false_negatives = (
        np.sum(confusion, axis=1) - true_positives
    )

    true_negatives = (
        total - true_positives - false_positives - false_negatives
    )

    return true_negatives / (true_negatives + false_positives)
