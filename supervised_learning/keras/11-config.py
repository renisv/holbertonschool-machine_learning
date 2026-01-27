#!/usr/bin/env python3
"""
11-config.py
Saves and loads a keras model configuration
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    network is the model whose configuration should be saved
    filename is the path of the file that the config should be saved to

    Returns: None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model from a JSON configuration file.

    filename is the path of the file containing the model configuration

    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        config = f.read()

    return K.models.model_from_json(config)
