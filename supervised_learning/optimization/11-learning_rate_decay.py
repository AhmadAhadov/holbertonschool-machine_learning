#!/usr/bin/env python3
'''module documented'''
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''learning rate decay'''
    time_interval = global_step // decay_step
    alpha = alpha / (1 + decay_rate * time_interval)
    return alpha
