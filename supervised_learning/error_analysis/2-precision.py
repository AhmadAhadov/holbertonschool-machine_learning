#!/usr/bin/env python3
'''module documented'''
import numpy as np


def precision(confusion):
    '''function documented'''
    res = []
    for i in range(len(confusion)):
        TP, FP, prec = 0, 0, 0
        for j in range(len(confusion[i])):
            if i == j:
                TP = confusion[j][i]
            else:
                FP += confusion[j][i]
        prec = TP / (TP + FP)
        res.append(prec)
    return np.asarray(res)
