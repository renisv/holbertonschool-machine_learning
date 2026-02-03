#!/usr/bin/env python3
"""4-moving_average.py"""

def moving_average(data, beta):
    """
    Calculates the bias-corrected weighted moving average of a data set.

    Args:
        data (list): list of numeric values
        beta (float): weight for the moving average (0 <= beta < 1 typically)

    Returns:
        list: bias-corrected moving averages
    """
    v = 0.0
    avgs = []

    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - (beta ** t))  # bias correction
        avgs.append(v_corrected)

    return avgs
