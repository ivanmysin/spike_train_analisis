# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import scipy.signal as sig
import mylib as lib
import time




dataDir = '/home/ivan/Data/speed_of_streem/test_7_layers/!Без пикротоксина/source_data/'
dataFile = "12-18-15_2_CL_скорость.wav"
data = read(dataDir + dataFile)
fd = data[0]

window_size = 16*3
winstart = 16*1

adiaphoriaFactor = 0.005
spikes_signal = data[1][0:1000000, 1].astype(float)
spikes_signal -= np.mean(spikes_signal)
#spikes_signal /= np.std(spikes_signal)
spikes_signal = 2*(spikes_signal - spikes_signal.min() ) / (spikes_signal.max() - spikes_signal.min() ) - 1

spikes_signal = lib.butter_bandpass_filter(spikes_signal, 500, fd/2-100, fd)

#loc_max_ind = np.asarray( sig.find_peaks_cwt(spikes_signal, np.arange(1, 80)), dtype=int )

loc_max_ind = np.asarray ( sig.argrelextrema(spikes_signal, np.greater, order=1) )
loc_max_vals = spikes_signal[loc_max_ind]

       
loc_min_ind = np.asarray ( sig.argrelextrema( spikes_signal, np.less, order=1) )
loc_min_vals = spikes_signal[loc_min_ind]
threshold = 4*np.median( np.abs(spikes_signal) / 0.6745 )
# 4.5*np.std(spikes_signal)
# 4.0 * sig.medfilt( np.abs(spikes_signal) / 0.6745, 32001 ) 


spikes_indexes = np.append(loc_max_ind[loc_max_vals >= threshold], \
                           loc_min_ind[loc_min_vals <= -threshold])
spikes_indexes = np.sort(spikes_indexes)   
spikes_indexes = spikes_indexes[ np.diff( np.append(0, spikes_indexes) ) > (fd * adiaphoriaFactor) ]


spikes = spikes_indexes/fd
print (spikes)

fig = plt.figure()
ax = fig.add_subplot(111)

for i in spikes_indexes:
    
    max_idx = np.argmax( spikes_signal[i:i+window_size] )
    l_ind = i - winstart + max_idx
    h_ind = l_ind + window_size
            
    if ( (l_ind < 0) or (h_ind>spikes_signal.size) ):
        continue
            
            
    examp = spikes_signal[l_ind:h_ind]
    ax.plot(examp)
    i += max_idx





times = np.linspace(0, spikes_signal.size/fd, spikes_signal.size)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, spikes_signal, "b")
ax.plot(times[[0, -1]], [threshold, threshold], "r")
ax.plot(times[[0, -1]], [-threshold, -threshold], "r")


print ("N of spikes: " + str(spikes.size) )
ax.scatter(spikes, np.ones(spikes.size)*0.8, s=50, color="blue", marker="*")
ax.set_xlim(220, 225)
ax.set_ylim(-1, 1)



'''
clear_spikes_signal = sig.medfilt(spikes_signal, 9)  
threshold = 4*np.std(clear_spikes_signal)
ax = fig.add_subplot(212)
ax.plot(times, clear_spikes_signal - spikes_signal, "b")
ax.plot([times[0], times[-1]], [threshold, threshold], "r")
ax.plot([times[0], times[-1]], [-threshold, -threshold], "r")
ax.set_xlim(times.min(), times.max())
ax.set_ylim(-1, 1)
'''

plt.show(block=False)
#plt.close(fig)

spikes = spikes_indexes/fd
print (spikes)




