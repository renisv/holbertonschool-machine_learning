#!/usr/bin/env python3
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Parameters:
        X (numpy.ndarray): shape (m, nx), where
            m is the number of data points
            nx is the number of features

    Returns:
        tuple: (mean, standard deviation) of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
