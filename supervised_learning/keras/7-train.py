#!/usr/bin/env python3
"""documented"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """documented"""
    if validation_data is None:
        history = network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle
        )
    else:
        if learning_rate_decay:
            learning_rate = K.optimizers.schedules.InverseTimeDecay(
                alpha,
                decay_steps=1,
                decay_rate=decay_rate,
                staircase=True
            )
        if early_stopping is True:
            early_stopping = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            history = network.fit(
                data,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                shuffle=shuffle,
                validation_data=validation_data,
                callbacks=[early_stopping]
            )
        else:
            history = network.fit(
                data,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                shuffle=shuffle,
                validation_data=validation_data
            )
    return history
