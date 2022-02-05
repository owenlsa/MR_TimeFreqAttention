'''
Gaussian Filter for dat2pic process.
'''

import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np


def MAG(N_signal):
    def filter(b, a, x):
        y = []
        y.append(b[0] * x[0])
        for i in range(1, len(x)):
            y.append(0)
            for j in range(len(b)):
                if i >= j:
                    y[i] = y[i] + b[j] * x[i - j]
                    j += 1
            for l in range(len(b)-1):
                if i > l:
                    y[i] = (y[i] - a[l+1] * y[i-l-1])
                    l += 1
            i += 1
        return y

    def MA(N_signal):
        b = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
        a = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        DN_signal = filter(b, a, N_signal)
        return DN_signal

    def Gaussianfilter(N_signal):
        r = 7
        sigma = 1
        GaussTemp = np.ones((1, r*2-1))
        for i in range(r*2-1):
            GaussTemp[0, i] = np.exp(-np.power((i+1-r), 2) /
                                     (2*np.power(sigma, 2)))/(sigma*np.sqrt(2*np.pi))
        DN_signal = N_signal
        for i in range(len(DN_signal)-r+1)[r-1:]:
            # for i = r : length(DN_signal)-r+1
            sigSlice = []
            for s in N_signal[i-r+1: i+r]:
                sigSlice.append(s)
            DN_signal[i] = np.dot(sigSlice, np.transpose(GaussTemp))
        return DN_signal

    DN1 = MA(N_signal)
    DN_signal = Gaussianfilter(DN1)
    return DN_signal
