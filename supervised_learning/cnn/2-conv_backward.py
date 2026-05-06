#!/usr/bin/env python3
'''module documented'''
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''
        dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the
        unactivated output of the convolutional layer

        A_prev is a numpy.ndarray of shape
        (m, h_prev, w_prev, c_prev) containing the
        output of the previous layer

        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution

        b is a numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution
        padding is a string that is either same or valid
        stride is a tuple of (sh, sw)
    '''
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]
    sh, sw = stride
    kh, kw = W.shape[0], W.shape[1]
    m = A_prev.shape[0]
    c_new = b.shape[3]
    if padding == "same":
        ph = int(((h_prev-1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev-1) * sw + kw - w_prev) / 2) + 1
    else:
        ph, pw = 0, 0
    output_h = int(1 + (h_prev + 2 * ph - kh) / sh)
    output_w = int(1 + (w_prev + 2 * pw - kw) / sw)
    output_c = W.shape[3]
    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (ph, ph),
                         (pw, pw), (0, 0)),
                        constant_values=0)
    dA_prev = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(m):
        for row in range(output_h):
            for col in range(output_w):
                for k in range(output_c):
                    patch = A_prev_pad[i, row * sh: row * sh + kh,
                                       col * sw: col * sw + kw, :]
                    dA = W[:, :, :, k] * dZ[i, row, col, k]
                    dA_prev[i, row * sh: row * sh + kh,
                            col * sw: col * sw + kw, :] += dA
                    dW[:, :, :, k] += (patch * dZ[i, row, col, k])
    if padding == "same":
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev
    return dA_prev, dW, db
