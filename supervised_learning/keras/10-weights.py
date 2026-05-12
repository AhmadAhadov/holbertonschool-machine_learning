#!/usr/bin/env python3
"""documented"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """documented"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """documented"""
    network.load_weights(filename)
    return None
