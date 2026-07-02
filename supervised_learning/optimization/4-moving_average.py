#!/usr/bin/env python3
'''module documented'''
import numpy as np


def moving_average(data, beta):
    '''method'''
    ewa = []
    temp = 0
    for i in range(1, len(data) + 1):
        temp = beta * temp + (1 - beta) * data[i - 1]
        temp_c = temp / (1 - beta ** i)
        ewa.append(temp_c)
    return ewa
