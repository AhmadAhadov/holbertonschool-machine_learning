#!/usr/bin/env python3
"""documented"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    network: the network model to make the prediction with
    data: the input data to make the prediction with
    verbose: boolean determining if output should be printed
             during the process

    Returns: the prediction for the data
    """
    # The predict method returns the raw output of the last layer
    prediction = network.predict(x=data, verbose=verbose)

    return prediction
