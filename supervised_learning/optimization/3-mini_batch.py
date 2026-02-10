#!/usr/bin/env python3
"""
3-mini_batch.py
Creates mini-batches for mini-batch gradient descent
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from X and Y.

    X: numpy.ndarray of shape (m, nx) containing input data
    Y: numpy.ndarray of shape (m, ny) containing labels
    batch_size: number of examples per batch
    Returns: list of tuples (X_batch, Y_batch)
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X_shuffled.shape[0]

    mini_batches = []
    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
