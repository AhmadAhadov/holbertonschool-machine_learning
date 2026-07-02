#!/usr/bin/env python3
'''module documented'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''Batch Normalization'''
    mean = np.mean(Z, axis=0)
    std = np.std(Z, axis=0)
    norm = (Z - mean) / (np.sqrt(std ** 2 + epsilon))
    return gamma * norm + beta
