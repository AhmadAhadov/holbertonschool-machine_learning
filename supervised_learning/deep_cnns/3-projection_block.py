#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    '''filters is a tuple or list containing
       F11, F3, F12, respectively:
       F11 - the number of filters in
       the first 1x1 convolution
       F3 - the number of filters in
       the 3x3 convolution
       F12 - the number of filters in the second
       1x1 convolution as well as the 1x1 convolution
       in the shortcut connection'''
    F11, F3, F12 = filters[0], filters[1], filters[2]
    initializer = K.initializers.HeNormal(seed=0)
    layer1 = K.layers.Conv2D(filters=F11,
                             kernel_size=(1, 1),
                             strides=(s, s),
                             padding='same',
                             kernel_initializer=initializer)(A_prev)
    layer1 = K.layers.BatchNormalization(axis=-1)(layer1)
    layer1 = K.layers.Activation('relu')(layer1)

    layer2 = K.layers.Conv2D(filters=F3,
                             kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=initializer)(layer1)
    layer2 = K.layers.BatchNormalization(axis=-1)(layer2)
    layer2 = K.layers.Activation('relu')(layer2)

    layer3 = K.layers.Conv2D(filters=F12,
                             kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=initializer)(layer2)
    layer3 = K.layers.BatchNormalization(axis=-1)(layer3)

    layer_s = K.layers.Conv2D(filters=F12,
                              kernel_size=(1, 1),
                              strides=(s, s),
                              padding="same",
                              kernel_initializer=initializer)(A_prev)
    layer_s = K.layers.BatchNormalization(axis=-1)(layer_s)
    res = K.layers.Add()([layer3, layer_s])
    res_A = K.layers.Activation('relu')(res)
    return res_A
