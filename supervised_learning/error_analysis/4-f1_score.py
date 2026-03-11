#!/usr/bin/env python3
'''module documented'''
import numpy as np


def f1_score(confusion):
    '''function documented'''
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    prec = precision(confusion)
    sens = sensitivity(confusion)
    res = ((2 * sens * prec) /
           (sens + prec))
    return res
