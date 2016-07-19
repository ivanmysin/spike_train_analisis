# -*- coding: utf-8 -*-
"""
stimulations descrimination
"""

import numpy as np
from scipy.io.wavfile import read
from scipy.io.matlab import savemat
from scipy.stats import variation as cv
from scipy import stats
import mylib as lib
import matplotlib.pyplot as plt
import os
from itertools import cycle
#######################################################################
def descriminate_stimulations(signal, fd, title, min_interstim_interval=0.25, threshold = 0.5):
    
    stims_find = np.ones(signal.shape[0], dtype=float)

    for ch_ind in range(signal.shape[1]):
        channel_signal = signal[:, ch_ind]
        
        stims_find *= np.abs(channel_signal)
        
    if (np.sum(stims_find) < 5):
        return []
    
    
#    t = np.linspace(0, stims_find.size/fd, stims_find.size)
#    sl = (t > 267 ) & (t < 271)
#    plt.figure()
#    plt.plot(t[sl], stims_find[sl])
    
    stims = np.argwhere(stims_find >= threshold)
    stims = stims.reshape(stims.size) / fd
    stims = stims[np.append([min_interstim_interval + 1], np.diff(stims)) > min_interstim_interval ]
    
    stims = 0.01 * ( np.round(stims*100) )
  
  
    stimulations = {}
    stims_intervals = np.diff(stims)
    plt.figure()
    plt.scatter(np.arange(len(stims)), stims)
    
    start_ind = 0
    end_ind = -1
    previous_interval = 2*stims_intervals[0]
    stims_in_series = 0
    counter = 1
    colors = cycle(["m", "c", "k", "y"])
    for idx, st_int in enumerate(stims_intervals):
        if ( np.abs(st_int - previous_interval) / st_int  > 0.2 or  (idx == stims_intervals.size - 1)):
            end_ind = idx


            if ( (end_ind - start_ind) > 5 and  cv(stims_intervals[start_ind:end_ind] < 0.2)):
            # ( st_int - np.mean( stims_intervals[start_ind:end_ind]) ) > 2*np.std(stims_intervals[start_ind:end_ind])
            # np.abs(st_int - next_interval) / st_int > 0.2):
            
                st = stims[start_ind:end_ind+1]
                stimulations[str(counter)] = st
                counter += 1
                stims_in_series += st.size
                
                frq_st = np.round( 1 / np.mean(np.diff(st)) )
                
                if (frq_st == 1):
                    color = "red"
                elif (frq_st == 3):
                    color = "green"
                else:
                    color = next(colors)


                plt.scatter(np.arange(start_ind, end_ind+1), st, color=color)
                
                
                print (frq_st, st.size, st[0], st[-1], st_int, previous_interval)
            #else:
            #    print ( stims_intervals[start_ind:end_ind] )
            start_ind = idx
        previous_interval = st_int
    #print (stims.size, stims_in_series, stims_in_series / stims.size)
    print ("Всего стимуляций в файле %i" % (counter-1))
    plt.title(title + " n stims = " + str(counter-1))
    plt.show(block=True)
    return stimulations
#######################################################################

main_path = '/home/ivan/Data/Ach_full/'
reading_path = main_path + 'source_data_simulations/'
saving_path = main_path + 'discrimination_simulation/'
for wavfile in sorted(os.listdir(reading_path)):
    if (wavfile[0] == "." or os.path.splitext(wavfile)[1] != ".wav"):
        continue
    saving_file = saving_path + os.path.splitext(wavfile)[0] + '_stims_descr.mat'
    if (os.path.isfile(saving_file)):
        continue
    wavcontent = read(reading_path + wavfile)
    fd = wavcontent[0]
    wavdata = wavcontent[1]
    
    wavdata = wavdata.astype(float) 
    wavdata =  2 * ( wavdata - wavdata.min() ) / (wavdata.max() - wavdata.min()) - 1  
    

    print ('###############################################')
    print (wavfile)
    stimulations = descriminate_stimulations(wavdata, fd, os.path.splitext(wavfile)[0])
        
    savemat(saving_file, stimulations)
