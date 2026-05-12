#!/usr/bin/env python3
"""documented"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """documented"""
    callbacks = []
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
            def learning_rate(epoch):
                return alpha / (1 + decay_rate * epoch)
            callbacks.append(K.callbacks.LearningRateScheduler(learning_rate, verbose=1))
            history = network.fit(
                data,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                shuffle=shuffle,
                validation_data=validation_data,
                callbacks=callbacks
            )
        if early_stopping is True:
            early_stopping = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(early_stopping)
            history = network.fit(
                data,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                shuffle=shuffle,
                validation_data=validation_data,
                callbacks=callbacks
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
