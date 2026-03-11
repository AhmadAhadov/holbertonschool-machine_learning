#!/usr/bin/env python3
'''module documented'''
import numpy as np


def sensitivity(confusion):
    '''module documented'''
    res = []
    for i in range(len(confusion)):
        TP, FN = 0, 0
        for j in range(len(confusion[i])):
            if i == j:
                TP = confusion[i][j]
            else:
                FN += confusion[i][j]
        sens = TP / (TP + FN)
        res.append(sens)
    return np.asarray(res)
