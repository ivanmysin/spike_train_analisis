# -*- coding: utf-8 -*-

a = ["a", "a", "d", "a", "b", "c"]
b = sorted(set(a))

print (b[1])

"""
import numpy as np

x = np.random.rand(10)

x_fft = np.fft.fft(x)

freqs = np.arange(0, x.size)#np.fft.fftfreq(x.size, 0.1)

x_dft = np.zeros(x.size)

for idx in range(x.size):
    x_dft[idx] = np.sum( x * np.cos( np.pi*2* freqs[idx] ) )

print (x_fft.real)
print (x_dft)

"""