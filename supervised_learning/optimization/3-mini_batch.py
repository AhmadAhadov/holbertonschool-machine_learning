#!/usr/bin/env python3
'''module documented'''
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    '''method'''
    X_shuffled, y_shuffled = shuffle_data(X, Y)
    mini_batches = []
    n = 1
    for i in range(0, len(X), batch_size):
        if batch_size * n > len(X):
            mini_batches.append((
                X_shuffled[i: len(X)],
                y_shuffled[i: len(X)]))
        else:
            mini_batches.append((
                X_shuffled[i: batch_size * n],
                y_shuffled[i: batch_size * n]))
        n += 1
    return mini_batches
