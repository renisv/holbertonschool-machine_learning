#!/usr/bin/env python3
"""
4-f1_score.py
Calculates the F1 score for each class
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class

    confusion: numpy.ndarray of shape (classes, classes)
    Returns: numpy.ndarray of shape (classes,)
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * sens) / (prec + sens)
