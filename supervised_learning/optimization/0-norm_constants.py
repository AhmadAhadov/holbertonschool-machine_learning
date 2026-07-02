#!/usr/bin/env python3
'''module documented'''
import numpy as np


def normalization_constants(X):
    '''normalization'''
    #  [10, 40, 145] age, weight, height
    m = np.mean(X, axis=0)
    std_ = np.std(X, axis=0)
    return m, std_
