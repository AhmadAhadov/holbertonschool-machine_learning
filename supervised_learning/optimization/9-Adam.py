#!/usr/bin/env python3
'''module documented'''
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''Adam optimization'''
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2
    v_corr = v / (1 - beta1 ** t)
    s_corr = s / (1 - beta2 ** t)
    var = var - alpha * v_corr / ((s_corr) ** 0.5 + epsilon)
    return var, v, s
