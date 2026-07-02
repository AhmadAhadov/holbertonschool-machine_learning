#!/usr/bin/env python3
'''module documented'''
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''gradient descent with momentum'''
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
