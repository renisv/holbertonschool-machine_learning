#!/usr/bin/env python3
"""
1-normalize.py
Normalizes (standardizes) a dataset using provided mean and standard deviation
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes a matrix X using mean m and standard deviation s

    X: numpy.ndarray of shape (d, nx)
    m: numpy.ndarray of shape (nx,) containing feature means
    s: numpy.ndarray of shape (nx,) containing feature standard deviations
    Returns: normalized X
    """
    return (X - m) / s
