#!/usr/bin/env python3
"""documented"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """documented"""
    # 784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(l2=lambtha)
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1-keep_prob)(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model
