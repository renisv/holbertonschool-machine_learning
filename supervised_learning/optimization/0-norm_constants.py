#!/usr/bin/env python3
import numpy as np

def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in X

    X: numpy.ndarray of shape (m, nx)
    Returns: mean, std (both numpy.ndarray of shape (nx,))
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)

    return m, s
