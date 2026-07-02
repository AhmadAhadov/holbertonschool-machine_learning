#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    '''gradient descent with momentum in keras'''
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
