# -*- coding: utf-8 -*-
"""
library function for spike train analisys 
"""
import numpy as np
from scipy import stats
import scipy.signal as sig
import matplotlib.pyplot as plt


###############################################################################
# function for processing
def get_rate_plot(spikes, step):
    
    time_bins = np.arange(0, np.max(spikes)+step, step)
    rate, _ = np.histogram(spikes, time_bins)    
    rate = rate.astype(dtype=float)/step

    return time_bins, rate
    
def get_hmsi(spikes, hmsi_bin, file_path, title=""):
    intervals = np.diff( spikes, n=1 )
    intervals = intervals[ np.abs(intervals - np.mean(intervals)) < 3*np.std(intervals) ]
    cv = stats.variation(intervals) #  np.sqrt(np.std(intervals)) / np.mean(intervals) #
    numBins = 100   
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    hmsi, bins, _ = ax.hist(intervals, numBins, color='green', alpha=0.8, normed=True)

    ax.set_title("HMSI " + title)
    
    
    
    fig.savefig(file_path, dpi=500)
    plt.show(block=False)
    plt.close(fig)
    
    #hmsi, bins = np.histogram(intervals, numBins, density=True)

    return bins, hmsi, cv
"""
def get_autororrelogram2(spikes, auc_bin, maxorder=100):
    if (maxorder<spikes.size):
        maxorder = spikes.size-1
    
    int_of_orders = np.array([], dtype=float)
    nbins = 100
    
    for order in range(1, maxorder+1) :
        intervals = spikes[order:-1] - spikes[0:-order-1]
        int_of_orders = np.append(int_of_orders, intervals)
    
    print ("Hello")
    auc, bins = np.histogram(int_of_orders, nbins)
    auc = auc.astype(dtype=float)/nbins
    
    step = bins[1]-bins[0]
    tau_mins = 0
    #tau_mins = get_tau_of_autocorreleogram(auc, step, side="low", npeaks=4)
    tau_maxs = get_tau_of_autocorreleogram(auc, step, side="top", npeaks=4)
    return bins, auc, tau_mins, tau_maxs
"""
def get_autororrelogram(spikes, auc_bin, maxorder=100):
    if (maxorder<spikes.size):
        maxorder = spikes.size-1
    intervals = spikes[maxorder:-1] - spikes[0:-maxorder-1]
    nbins = 150 # np.max(intervals)/auc_bin + 1

    auc, bins = np.histogram(intervals, nbins)
    auc = auc.astype(dtype=float)
    for order in range(maxorder-1) :
        order+=1
        intervals = spikes[order:-1] - spikes[0:-order-1]
        auc_tmp, _ = np.histogram(intervals, bins)
        auc += auc_tmp.astype(float) #/auc_bin
    
    auc = auc.astype(dtype=float)/maxorder
    step = bins[1]-bins[0]
    tau_mins = 0
    #tau_mins = get_tau_of_autocorreleogram(auc, step, side="low", npeaks=4)
    tau_maxs = get_tau_of_autocorreleogram(auc, step, side="top", npeaks=4)
    del(spikes)
    return bins[0:-2], auc[0:-2], tau_mins, tau_maxs
    #time, autocorrelogram_plot = make_as_rabbit(bins, auc) 
    #return time, autocorrelogram_plot    
def get_tau_of_autocorreleogram(auc, step, side="top", npeaks=5):
    if (side == "top"):
        extrenums_ind = np.asarray ( sig.argrelextrema(auc, np.greater), order=3 )
    else:
        extrenums_ind = np.asarray ( sig.argrelextrema(auc, np.less) )
    if (extrenums_ind.size < 2):
        return 0
    extrenums_ind = extrenums_ind[0, 0:npeaks-1]
    extrenums_vals = auc[extrenums_ind] 
    x_reg = extrenums_ind * step
    y_reg = -1 * np.log( extrenums_vals )
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_reg, y_reg)
    del(auc)
    return slope

    
def get_neuron_spectra(autocorrelegram, auc_bin):
    Ns = autocorrelegram.size
    spectra = np.fft.fft(autocorrelegram)
    frq = np.fft.fftfreq(Ns, auc_bin)
    inx = (frq>0)
    frq = frq[inx]
    spectra = np.abs(spectra[inx]) / Ns
    mode_ind = np.argmax(spectra)
    del(autocorrelegram)
    return frq, spectra, frq[mode_ind]
