#!/usr/bin/env python3
"""
0-create_confusion.py
Creates a confusion matrix from one-hot encoded labels and predictions
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    labels: one-hot numpy.ndarray of shape (m, classes)
    logits: one-hot numpy.ndarray of shape (m, classes)
    Returns: confusion matrix of shape (classes, classes)
    """
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)

    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1

    return confusion
