#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    '''RMSProp in keras'''
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
