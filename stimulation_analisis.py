# -*- coding: utf-8 -*-
"""
script for stimulation analisis
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os

###############################################################################
def get_psh(spikes, stimulations, saving_file=None, period4freq=10):
    mean_interval = np.mean( np.diff(stimulations) )
    spikes_in_stim = np.empty((0), dtype=float)
    
    starts_ind = np.empty((stimulations.size), dtype=int)
    ends_ind = np.empty((stimulations.size), dtype=int)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    latent_period = 0
    background = 0
    answer = 0
    for idx, stim in enumerate(stimulations):
        start = stim - 0.1*mean_interval
        end = stim + 0.9*mean_interval
        
        sp = spikes[ (spikes >= start) & (spikes <= end) ] - stim  
        if (sp.size < 5):
            continue
        starts_ind[idx] = spikes_in_stim.size
        ends_ind[idx] = spikes_in_stim.size + sp.size - 1
        spikes_in_stim = np.append(spikes_in_stim, sp)
        
        ax.scatter(sp, np.zeros(sp.size)-idx-1, s=10)
        
        latent_period += np.min( sp[sp>0] )
        background += np.sum( sp < 0 )
        answer += np.sum( sp > 0 )
        
    if (spikes_in_stim.size < 10):
        plt.close(fig)
        return 0, 0, 0, 0, 0, 0
        
        
    latent_period /= stimulations.size
    background = background / stimulations.size / (0.1*mean_interval)
    answer = answer / stimulations.size / (0.9*mean_interval)
    
        
    top_PSH, bins = np.histogram(spikes_in_stim, 20) 
    top_PSH = np.append(top_PSH[0], top_PSH)
    
    
    ax.step(bins, top_PSH)
    ax.set_xlim(-0.1*mean_interval, 0.9*mean_interval)
    ax.set_ylim(-idx-1, 1.2*np.max(top_PSH) )
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if not(type(saving_file) is None):
        fig.savefig(saving_file, dpi=200)
    plt.show(block=False)
    plt.close(fig)
    
    
    stimulation_frequency = 1 / mean_interval
    
    if (period4freq > 0):
        start = stimulations[0] - period4freq
        end = stimulations[0]
        sp = spikes[ (spikes >= start) & (spikes <= end) ]
        frq_before_stim = sp.size / period4freq
    
        start = stimulations[-1] + mean_interval
        end = start + period4freq
        sp = spikes[ (spikes >= start) & (spikes <= end) ]
        frq_after_stim = sp.size / period4freq
    else:
        frq_before_stim = 'inf'
        frq_after_stim = 'inf'
    return latent_period, background, answer, stimulation_frequency, frq_before_stim, frq_after_stim
###############################################################################

main_path = "/home/ivan/Data/Ach_full/"

spikes_dir = main_path + "processing_stimulations/discriminated_spikes/"
stimulation_dir = main_path + "discrimination_simulation/"
saving_dir = main_path + "processing_stimulations/induced_activity/"


if not( os.path.isdir(saving_dir) ):
    os.mkdir(saving_dir)

for spikes_file in sorted( os.listdir(spikes_dir) ):
    
    if (spikes_file[0] == '.' or os.path.splitext(spikes_file)[1] != '.mat'):
        continue

    stimulation_file = spikes_file[0:3] + "_stims_descr.mat"
    
    print (stimulation_file)
    if not( os.path.isfile(stimulation_dir + stimulation_file) ):
        continue
    spikes = io.loadmat(spikes_dir + spikes_file)
    
    stimulations = io.loadmat(stimulation_dir + stimulation_file)
    
    for key, sp in spikes.items():
        if not (type(sp) is np.ndarray):
            continue
        sp = sp[0, :]
        
        neuron_dir = saving_dir + spikes_file[0:3]
        neuron_dir += "_channel_" + spikes_file.split("discr_channel_")[1][0]
        neuron_dir += "_neuron_" + key + "/"
        
        
        if not( os.path.isdir(neuron_dir) ):
            os.mkdir(neuron_dir)
        
        statistics_by_one_neuron = """Stimulations frequency (Hz) \t First stimulation in seria (sec) \t Last stimulation in seria(sec) \t Background (sp/sec) \t Mean latent period (sec) \t Response(sp/sec)"""    
        statistics_by_one_neuron += "\t Spike frequency before stim (sp/sec) \t Spike frequency after stim (sp/sec) \n"
        ##################################################################
        for stims in stimulations.values():
            if not (type(stims) is np.ndarray):
                continue
            stims = stims.reshape(stims.size)
            
            saving_file = neuron_dir + str(np.round(stims.mean())) + ".png"
            
            latent_period, background, response, stimulation_frequency, frq_before_stim, frq_after_stim = get_psh(sp, stims, saving_file)
            #print (latent_period, background, answer)
            statistics_by_one_neuron += "%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n" \
                                       % (stimulation_frequency, stims[0], stims[-1], background, latent_period, response, frq_before_stim, frq_after_stim)
            statfile = open(neuron_dir + "stat.txt", "w")
            statfile.write(statistics_by_one_neuron)
            statfile.close()
