#!/usr/bin/env python3
'''module documented'''
import numpy as np


def shuffle_data(X, Y):
    '''function'''
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = Y[indices]
    return X_shuffled, y_shuffled
