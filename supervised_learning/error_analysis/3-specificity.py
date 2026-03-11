#!/usr/bin/env python3
'''module documented'''
import numpy as np


def specificity(confusion):
    '''function documented'''
    res = []
    total = confusion.sum()
    for i in range(len(confusion)):
        TP = confusion[i, i]
        FP = confusion[:, i].sum() - TP
        FN = confusion[i, :].sum() - TP
        TN = total - TP - FP - FN
        spec = TN / (TN + FP)
        res.append(spec)
    return np.asarray(res)
