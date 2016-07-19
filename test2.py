# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import mannwhitneyu

a = np.random.randn(10, 10)
b = np.linalg.inv(a)

"""
import numpy as np
from scipy.io.wavfile import read, write
import os
origin_path = '/home/ivan/Data/Ach/origin_data/not cutted/'
target_path = '/home/ivan/Data/Ach/origin_data/'

for wavfile in os.listdir(origin_path):
    if ("_2" in wavfile or wavfile[0] == "."):
        continue
    wav1 = read(origin_path + wavfile)
    
    wavfile2 = os.path.splitext(wavfile)[0] + "_2.wav"
    if not(os.path.isfile(origin_path + wavfile2)):
        continue
    wav2 = read(origin_path + wavfile2)
    
    wav_merged = np.append(wav1[1], wav2[1], axis=0)
    
    write(target_path + wavfile, wav1[0], wav_merged)
"""

"""
import numpy as np

x = np.random.rand(10)

x_fft = np.fft.fft(x)

freqs = 2*np.pi*np.linspace(0, 1, x.size)#np.fft.fftfreq(x.size, 0.1)

x_dft = np.zeros_like(x)

for idx in range(x.size):
    x_dft[idx] = np.sum( x * np.cos( freqs[idx]/x.size ) )

print (x_fft.real)
print (x_dft)



N = x.size             # length of test data vector
data = x               # test data
X = np.zeros(N)        # pre-allocate result
for k in range(N):
    for n in range(N):
        X[k] += data[n]*np.cos(2*np.pi *n*k/N)
        #print (n*k/N)
    
print (X)
"""
