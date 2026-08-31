#!/usr/bin/env python3
"""Simple Policy function."""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix."""
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)
