#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    '''func builds the ResNet-50 architecture'''
    initializer = K.initializers.HeNormal(seed=0)
    input = K.Input(shape=(224, 224, 3))
    X = K.layers.Conv2D(
                        filters=64,
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=initializer)(input)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding="same")(X)
    filters = (64, 64, 256)
    proj_b1 = projection_block(X,
                               filters,
                               s=(1, 1))
    filters = (64, 64, 256)
    id_block1 = identity_block(proj_b1,
                               filters)
    id_block2 = identity_block(id_block1,
                               filters)

    filters = (128, 128, 512)
    proj_b2 = projection_block(id_block2, filters)

    filters = (128, 128, 512)
    id_block2 = identity_block(proj_b2, filters)
    id_block2 = identity_block(id_block2, filters)
    id_block2 = identity_block(id_block2, filters)

    filters = (256, 256, 1024)
    proj_b3 = projection_block(id_block2, filters)

    filters = (256, 256, 1024)
    id_block3 = identity_block(proj_b3, filters)
    id_block3 = identity_block(id_block3, filters)
    id_block3 = identity_block(id_block3, filters)
    id_block3 = identity_block(id_block3, filters)
    id_block3 = identity_block(id_block3, filters)

    filters = (512, 512, 2048)
    proj_b4 = projection_block(id_block3, filters)
    id_block4 = identity_block(proj_b4, filters)
    id_block4 = identity_block(id_block4, filters)
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                         padding='valid')(id_block4)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=initializer)(avg_pool)
    return K.Model(inputs=input, outputs=softmax)
