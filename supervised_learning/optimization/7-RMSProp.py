#!/usr/bin/env python3
'''module documented'''
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''gradient descent with momentum'''
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * (grad / (s ** 0.5 + epsilon))
    return var, s
