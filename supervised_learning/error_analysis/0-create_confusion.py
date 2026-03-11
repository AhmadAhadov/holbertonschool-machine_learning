#!/usr/bin/env python3
'''module documented'''
import numpy as np


def create_confusion_matrix(labels, logits):
    '''function documented'''
    res = np.zeros((labels.shape[1], logits.shape[1]))
    for i in range(len(labels)):
        row = np.where(labels[i] == 1)[0][0]
        col = np.where(logits[i] == 1)[0][0]
        res[row][col] += 1
    return res
