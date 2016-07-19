# -*- coding: utf-8 -*-
"""
function of descrimination of stimulation (old variant)
"""

import numpy as np
import mylib as lib
from scipy import stats

def descriminate_stimulations(spikes_signals, fd, minimal_interval = 0.05, ratio_coef = 5):

    stims_find = np.ones(spikes_signals.shape[0], dtype=float)
    stims_series = {}
    for ch_ind in range(spikes_signals.shape[1]):
        original_spikes_signal = spikes_signals[:, ch_ind]
        spikes_signal = 2*( original_spikes_signal - original_spikes_signal.min() ) / (original_spikes_signal.max() - original_spikes_signal.min() ) - 1
     
        spikes_signal = lib.butter_bandpass_filter(spikes_signal, 500, fd/2-100, fd)
     
        threshold = 3*np.median( np.abs(spikes_signal) / 0.6745 )
        spikes_signal[(spikes_signal>-threshold)&(spikes_signal<threshold)] = 0
        stims_find *= spikes_signal
        
    if (np.sum(stims_find) < 5):
        return False

    stims = np.argwhere(stims_find > 0) 
    #lib.get_argextremums(stims_find)
    #np.asarray ( sig.argrelextrema(stims_find, np.greater, order=1) )
    stims = (stims/fd).astype(float)
    stims_diff = stims[1:] - stims[:-1]
    
    stims_diff = np.append(stims_diff, stims_diff.max() )
    stims = stims[ stims_diff>minimal_interval ]
    stims_diff = stims[1:] - stims[:-1]
    
    stims_diff = 0.1 *( np.round(stims_diff*10) )
    mode_diff = stats.mode(stims_diff)
    
    
    

    counter = 0
    for idx, mode in enumerate(mode_diff[0]):
        start_ind = -1
        end_ind =-1
        for ind in range(stims.size):

            if (ind == 0 and (stims[ind+1] - stims[ind]) < ratio_coef*mode ):
                start_ind = ind
                continue
            
            if ( ind == stims.size-1 and (stims[ind] - stims[ind-1]) < ratio_coef*mode):
                end_ind = ind
                if (start_ind != -1):
                    st = stims[start_ind:end_ind]
                    if ( st.size>3 and stats.variation(np.diff(st)) < 0.2): 
                        stims_series[str(counter)] = st
                        counter +=1
                start_ind = -1
                continue
            if ( ind == stims.size-1): continue
            
            if ( (stims[ind] - stims[ind-1]) > ratio_coef*mode and (stims[ind+1] - stims[ind]) < ratio_coef*mode ):
                start_ind = ind
                continue
            
            if ( (stims[ind] - stims[ind-1]) < ratio_coef*mode and (stims[ind+1] - stims[ind]) > ratio_coef*mode ):
                end_ind = ind
                if (start_ind != -1):
                    st = stims[start_ind:end_ind]
                    if ( st.size>3 and stats.variation(np.diff(st)) < 0.2):
                        stims_series[str(counter)] = st
                        counter +=1
                start_ind = -1
                continue
    
    return stims_series

