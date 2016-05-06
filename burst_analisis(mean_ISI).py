# -*- coding: utf-8 -*-
"""

Detection of bursts in neuronal spike trains by the mean
inter-spike interval method

"""

import numpy as np
from scipy import io
from sklearn.cluster import MeanShift
import mylib
import matplotlib.pyplot as plt
import os
from itertools import cycle

###############################################################################
def find_bursts(isi, reriod=31):
    mmed_signal = moving_percentile(isi, reriod, 99, remove_center = True) 
    mmed_background = moving_percentile(isi, reriod, 1, remove_center = False) 
    
    first_spikes = np.empty((0), dtype=int)
    last_spikes = np.empty((0), dtype=int)
    
    for idx in range(isi.size):
        
        if (idx == 0 and mmed_background[idx] < mmed_signal[idx]):
           first_spikes = np.append(first_spikes, idx)
           continue
           
        if (idx == isi.size-1 and mmed_background[idx] < mmed_signal[idx]):
            last_spikes = np.append(last_spikes, idx+1)
            continue
        
        if ( mmed_background[idx-1] > mmed_signal[idx-1] and mmed_background[idx] < mmed_signal[idx] ):
            first_spikes = np.append(first_spikes, idx)
            continue
        
        if ( mmed_background[idx-1] < mmed_signal[idx-1] and mmed_background[idx] > mmed_signal[idx] ):
            last_spikes = np.append(last_spikes, idx)
            continue
        
    if (last_spikes.size < first_spikes.size):
        last_spikes = np.append(last_spikes, idx)
    
    if (last_spikes.size > first_spikes.size):
        first_spikes = np.append(0, first_spikes)        
        
    level = np.mean(isi) 

    print ( np.sum(last_spikes < first_spikes) )
    sl = isi[ last_spikes[:-1] ] > level
    sl_last = np.append(sl, True)
    sl_first = np.append(True, sl)
    
    last_spikes = last_spikes[sl_last]
    first_spikes = first_spikes[sl_first]
    #spikes_number = last_spikes - first_spikes + 1
    
    
    return first_spikes, last_spikes
###############################################################################
def moving_std(x, ma, n, mode="simple"):
    x = np.asarray(x)
    if mode=='simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))
    x = (x - ma)**2
    #weights /= weights.sum()

    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = np.sqrt( a[n] )
    a /= weights.sum()
    return a
###############################################################################
def moving_percentile(x, period, percent, remove_center = True):
    m_percenle = np.empty(x.size, dtype=float)
    
    for idx in range(x.size):
        start = idx - period//2
        end = start + period 
        if start<0:
            start = 0
        if end > x.size:
            end = x.size
        sl = np.zeros_like(x).astype(bool) 
        sl[start:end] = True
        if (remove_center):
            sl[idx] = False
        m_percenle[idx] = np.percentile(x[sl], 95)
    return m_percenle

###############################################################################
files = [
    {"filename":"Test_02-17-2016_1_пикротоксин_стимуляция_скорость", "channel": "1"}, 
    {"filename":"Test_02-17-2016_1_пикротоксин_стимуляция_скорость_обрез", "channel": "1"}, 
    {"filename":"Test_02-17-2016_1_пикротоксин_стимуляция_скорость_обрез", "channel": "1"}, 
    {"filename":"Изм_протока_25-12-15_2", "channel": "1"}, 
    {"filename":"Крыса_изм_прот_02-02-16_2", "channel": "2"}, 
    {"filename":"Крыса_изм_прот_19-12-15_1", "channel": "1"}, 
    {"filename":"Крыса_изменение_протока18-12-15", "channel": "2"}, 
    {"filename":"Крыса_изменение_протока_оп2_18-12-15", "channel": "1"}, 
    {"filename":"Крыса_изменение_протока_оп2_18-12-15", "channel": "2"}, 
]

path = "./discrimination_results/"

result_path = "./burst_result/burst_detection/"

for spike_file in files:
    spike_path = path + "discr_channel_" + spike_file["channel"] + "_"+ spike_file["filename"]
    data = io.loadmat(spike_path)


    
    for spike_train_number in data.keys():
        if (type(data[spike_train_number]) is not np.ndarray):
            continue
        spike_train = data[spike_train_number][0,:]
        isi = np.diff(spike_train)
        
        """
        first_spikes, last_spikes = find_bursts(isi)

        fd = 0.01
        first_spikes = spike_train[first_spikes.astype(int)]
        last_spikes = spike_train[last_spikes.astype(int)]
        
        spike = np.zeros( spike_train[-1]/fd + 1 )
        spike[ np.floor(spike_train/fd).astype(int) ] = 1
        time = np.linspace(0, spike_train[-1], spike.size)
        
        saving_path = result_path + "discr_channel_" + spike_file["channel"] + "_"+ spike_file["filename"]
        if not( os.path.isdir(saving_path) ):
            os.mkdir(saving_path)  
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sp = np.empty((spike_train.size, 2), dtype=float)
        sp[:, 0] = np.append(np.max(isi), isi)
        sp[:, 1] = np.append(isi, np.max(isi) )
                
        cl = MeanShift().fit(sp) #mylib.XMeans().
        ax.set_title("N of clusters = %i" % cl.cluster_centers_.shape[0])        
        
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for idx in range(cl.cluster_centers_.shape[0]):
            ax.scatter(sp[idx==cl.labels_, 0], sp[idx==cl.labels_, 1], s=30, c=next(colors))
        
        

        """
        ax.plot(time, spike, linewidth=3)
        ax.scatter(first_spikes, np.ones(first_spikes.size), s=70, color="red")
        ax.scatter(last_spikes, np.ones(last_spikes.size), s=70, color="green")
        ax.set_ylim(-0.5, 1.5)
        for start in np.arange(0, spike_train[-1] + 1, 20):
            ax.set_xlim(start, start+30)
            fig.savefig(saving_path + "/" + str(start) + ".png", dpi=200)
        plt.show(fig)
        
        plt.close(fig)
        """
        

    #break


###############################################################################
"""        
x = np.linspace(1, isi.size, isi.size)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, isi, "b", x, mmed_signal, "r", x, mmed_background, "g" , linewidth=3) 
ax.set_xlim(0, 100)
#plt.plot([0, isi.size], [ml, ml], "r")
"""


        

