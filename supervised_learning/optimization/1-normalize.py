#!/usr/bin/env python3
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
        X (numpy.ndarray): shape (d, nx)
            d is the number of data points
            nx is the number of features
        m (numpy.ndarray): shape (nx,), mean of each feature
        s (numpy.ndarray): shape (nx,), standard deviation of each feature

    Returns:
        numpy.ndarray: normalized X
    """
    return (X - m) / s
