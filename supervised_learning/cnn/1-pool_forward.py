#!/usr/bin/env python3
'''module documented'''
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        kernel_shape is a tuple of (kh, kw) containing
        the size of the kernel for the pooling
        stride is a tuple of (sh, sw)
    '''
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]
    sh, sw = stride
    kh, kw = kernel_shape
    m = A_prev.shape[0]
    c_prev = A_prev.shape[3]
    output_h = int(1 + (h_prev - kh) / sh)
    output_w = int(1 + (w_prev - kw) / sw)
    output = np.zeros((m, output_h, output_w, c_prev))
    for row in range(output_h):
        for col in range(output_w):
            patch = A_prev[:, row * sh: row * sh + kh,
                           col * sw: col * sw + kw, :]
            if mode == "max":
                output[:, row, col, :] = (np.max(patch, axis=(1, 2)))
            else:
                output[:, row, col, :] = (np.mean(patch, axis=(1, 2)))
    return output
