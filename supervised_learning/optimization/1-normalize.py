#!/usr/bin/env python3
'''module documented'''
import numpy as np


def normalize(X, m, s):
    '''normalization'''
    #  [10, 40, 145] age, weight, height
    return (X - m) / s
