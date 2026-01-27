#!/usr/bin/env python3
"""
3-one_hot.py
Converts a label vector into a one-hot matrix
"""

import numpy as np


def one_hot(labels, classes=None):
    if classes is None:
        classes = np.max(labels) + 1

    one_hot_matrix = np.zeros((labels.shape[0], classes))
    one_hot_matrix[np.arange(labels.shape[0]), labels] = 1

    return one_hot_matrix
