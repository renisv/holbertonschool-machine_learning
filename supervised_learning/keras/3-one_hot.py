#!/usr/bin/env python3
"""
3-one_hot.py
Converts a label vector into a one-hot matrix
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    if classes is None:
        return K.utils.to_categorical(labels)
    return K.utils.to_categorical(labels, classes)
