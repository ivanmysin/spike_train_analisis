# -*- coding: utf-8 -*-
"""
script for stimulation analisis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os

###############################################################################
def get_psh(spikes, stimulations, saving_file=None):
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
        background += np.sum( sp<0 )
        answer += np.sum( sp>0 )
    
    if (spikes_in_stim.size < 10):
        return 0, 0, 0
        
        
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
    plt.show(fig)
    plt.close(fig)
    return latent_period, background, answer
###############################################################################

main_path = "/home/ivan/Data/speed_of_streem/!С пикротоксином и стимуляцией/"

spikes_dir = main_path + "processing/discriminated_spikes/"
stimulation_dir = main_path + "processing/discriminated_stimulations/"
saving_dir = main_path + "processing/induced_activity/"


for spikes_file in os.listdir(spikes_dir):

    stimulation_file = "discr_stimulations_" + spikes_file.split("discr_channel_")[1][2:]
    
    #print (stimulation_file)
    if not( os.path.isfile(stimulation_dir + stimulation_file) ):
        continue
    spikes = io.loadmat(spikes_dir + spikes_file)
    
    stimulations = io.loadmat(stimulation_dir + stimulation_file)
    
    for key, sp in spikes.items():
        if not (type(sp) is np.ndarray):
            continue
        sp = sp[0, :]
        
        neuron_dir = saving_dir + spikes_file.split("discr_channel_")[1][2:]
        neuron_dir = neuron_dir[0:-4]
        neuron_dir += "_channel_" + spikes_file.split("discr_channel_")[1][0]
        neuron_dir += "_neuron_" + key + "/"
        
        if not( os.path.isdir(neuron_dir) ):
            os.mkdir(neuron_dir)
        
        statistics_by_one_neuron = """
First stimulation in seria (sec) \t Last stimulation in seria(sec) \t Background (sp/sec) \t Mean latent period (sec) \t Response(sp/sec) \n     
"""
        ##################################################################
        for stims in stimulations.values():
            if not (type(stims) is np.ndarray):
                continue
            stims = stims[0,:]
            
            saving_file = neuron_dir + str(np.round(stims.mean())) + ".png"
            
            latent_period, background, response = get_psh(sp, stims, saving_file)
            #print (latent_period, background, answer)
            statistics_by_one_neuron += """
%f \t %f \t %f \t %f \t %f \n     
            """ % (stims[0], stims[-1], background, latent_period, response)
            statfile = open(neuron_dir + "stat.txt", "w")
            statfile.write(statistics_by_one_neuron)
            statfile.close()
