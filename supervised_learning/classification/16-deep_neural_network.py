#!/usr/bin/env python3
"""
16-deep_neural_network.py

Defines a DeepNeuralNetwork class that performs binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        Initialize a deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List of positive integers where each element is the
                number of nodes in that layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is not a positive integer.
            TypeError: If layers is not a list of positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx
        layer = 1
        for nodes in layers:
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.weights["W{}".format(layer)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.weights["b{}".format(layer)] = np.zeros((nodes, 1))
            prev = nodes
            layer += 1
