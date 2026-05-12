#!/usr/bin/env python3
"""documented"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network and returns the loss and accuracy.

    network: the model to test
    data: the input data to test with
    labels: the one-hot labels for the data
    verbose: boolean determining if output should be printed

    Returns: (loss, accuracy)
    """
    # evaluate returns a list: [loss, accuracy, ...]
    # based on the metrics used during model.compile()
    metrics = network.evaluate(x=data, y=labels, verbose=verbose)

    return metrics