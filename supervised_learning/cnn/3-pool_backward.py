#!/usr/bin/env python3
'''module documented'''
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
        dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the
        output of the pooling layer

        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the output of the previous layer

        kernel_shape is a tuple of (kh, kw) containing
        the size of the kernel for the pooling

        stride is a tuple of (sh, sw)
    '''
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]
    sh, sw = stride
    kh, kw = kernel_shape
    m = A_prev.shape[0]
    c_new = dA.shape[3]
    output_h = int(1 + (h_prev - kh) / sh)
    output_w = int(1 + (w_prev - kw) / sw)
    output = np.zeros((m, output_h, output_w, c_new))
    dA_prev = np.zeros_like((A_prev))
    for i in range(m):
        for row in range(output_h):
            for col in range(output_w):
                for k in range(c_new):
                    patch = A_prev[i, row * sh: row * sh + kh,
                                   col * sw: col * sw + kw, k]
                    da = dA[i, row, col, k]
                    if mode == "max":
                        mask = (patch == np.max(patch))
                        dA_prev[i, row * sh: row * sh + kh,
                                col * sw: col * sw + kw, k] += mask * da
                    else:
                        res = da / (kh * kw) * np.ones(kernel_shape)
                        dA_prev[i, row * sh: row * sh + kh,
                                col * sw: col * sw + kw, k] += res
    return dA_prev
