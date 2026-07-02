#!/usr/bin/env python3
'''module documented'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''batch normalization'''
    beta = tf.keras.initializers.Zeros()
    gamma = tf.keras.initializers.Ones()
    initializer = tf.keras.initializers.VarianceScaling(
        mode="fan_avg"
        )
    Z = tf.keras.layers.Dense(n,
                              kernel_initializer=initializer
                              )(prev)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-7,
                                            beta_initializer=beta,
                                            gamma_initializer=gamma)
    Z_norm = bn(Z, training=True)
    activated = activation(Z_norm)
    return activated
