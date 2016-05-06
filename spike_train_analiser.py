# -*- coding: utf-8 -*-
"""
script for spike train analisis
"""
import os
import numpy as np
from scipy import io
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
    
def get_hmsi(spikes, hmsi_bin, file_path):
    intervals = np.diff( spikes, n=1 )
    intervals = intervals[ np.abs(intervals - np.mean(intervals)) < 3*np.std(intervals) ]
    cv = stats.variation(intervals) #  np.sqrt(np.std(intervals)) / np.mean(intervals) #
    numBins = 100    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    hmsi, bins, _ = ax.hist(intervals, numBins, color='green', alpha=0.8, normed=True)

    ax.set_title("HMSI")
    
    
    
    fig.savefig(file_path, dpi=500)
    plt.show(fig)
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

###############################################################################
    
import getbounds 
main_path = '/home/ivan/Data/speed_of_streem/!Без пикротоксина/'
#"/home/ivan/Data/Ach/"
loadDir = main_path + "processing/discriminated_spikes/"
#"/home/ivan/Data/speed_os_streem/Крысы пикротоксин/discriminated_spikes/" # "./discrimination_results/"
boundsdir = main_path + "bounds/"
#"/home/ivan/Data/speed_os_streem/Крысы пикротоксин/bounds/"
saveDir = main_path + "processing/statistics/"
if not ( os.path.isdir(saveDir) ):
    os.mkdir(saveDir)
# "/home/ivan/Data/speed_os_streem/Крысы пикротоксин/whole_stats/"#"./figures/"
binForRatePlot = 10  
hmsi_bin = 0.005 # 100 ms


statisticsByAllNeurons = "file name \t" 
statisticsByAllNeurons += "N of spikes \t mean frequency \t CV \t tau (maximums) \t mode frequency \t"
statisticsByAllNeurons += "N of spikes \t mean frequency \t CV \t tau (maximums) \t mode frequency \t"
statisticsByAllNeurons += "N of spikes \t mean frequency \t CV \t tau (maximums) \t mode frequency \n"


for dataFile in os.listdir(loadDir):
    print (dataFile)
    spikes = io.loadmat(loadDir + dataFile)
    channelDir = dataFile.split("_discr_")[0] + dataFile.split("_discr")[1]
    channelDir = os.path.splitext(channelDir)[0]
    
    csvboundsfile = dataFile.split("discr_channel_")[0]

    channel_number = int(channelDir[-1])
    
    csvboundsfile = csvboundsfile.split("CL")[0] + "CL.csv"
    if not ( os.path.isfile(boundsdir + csvboundsfile) ):
        continue
    bounds = getbounds.get_bounds(boundsdir + csvboundsfile)
        
    reportofsoursefile = channelDir
    for key in spikes.keys():
                
        
        if not (type(spikes[key]) is np.ndarray): continue
        
        sp = spikes[key][0]
     
        
        if (sp.size < 20): continue
        neuronDir = saveDir + channelDir + '_neuron_' + key + "/"

        if not ( os.path.isdir(neuronDir) ):
            os.mkdir(neuronDir)
        statisticsByAllNeurons += reportofsoursefile + '_neuron_' + key + "\t"
        
        
        time_steps, rate_plot = get_rate_plot(sp, binForRatePlot)
        figOfRatePlot, axOfRatePlot = plt.subplots() 
        axOfRatePlot.step(time_steps[0:-1], rate_plot)
        axOfRatePlot.set_title("Rate plot")
        axOfRatePlot.set_xlabel("time, sec")
        axOfRatePlot.set_ylabel("spike rate, sp/sec")
        axOfRatePlot.set_xlim(0, time_steps.max())
        if (rate_plot.max() > 100):
            axOfRatePlot.set_ylim(0, 30)
        else:
            axOfRatePlot.set_ylim(0, 1.2*rate_plot.max())
        figOfRatePlot.savefig(neuronDir + "Rate_plot", dpi=500)
    
        statisticsByNeuron = dataFile + """\n
        Comment \t start of period \t end of period \t N of spikes \t mean frequency \t CV \t tau (maximums) \t mode frequency \n
        """        
        
        
        for idx, bound in enumerate(bounds):
            if (bound["channel"] != channel_number):
                continue
            
            numberOfBound = str(idx + 1)
            condition_idx = np.logical_and( (sp>=bound["low_bound"]), (sp<=bound["upper_bound"]) )
            sp_in_bound = np.sort( sp[condition_idx] )
            if (sp_in_bound.size < 10):
                continue
            
             
            hmsi_file = neuronDir + "Hmsi_" + numberOfBound
            hmsi_bins, hmsi, cv = get_hmsi(sp_in_bound, hmsi_bin, hmsi_file)
            
            auc_times, auc, tau_mins, tau_maxs  = get_autororrelogram(sp_in_bound, hmsi_bin)
            figOfAuc, axOfAuc = plt.subplots() 
            
            if ( np.sum(auc) > 0.001):
                axOfAuc.step(auc_times[0:-1], auc)
                axOfAuc.set_title("Autocorrelelogram of " + numberOfBound)      
                axOfAuc.set_ylim(0, 1.2*np.max(auc) )
                figOfAuc.savefig(neuronDir + "Autocorrelelogram_" + numberOfBound, dpi=500)
               
            frq, spectra, modeFr = get_neuron_spectra(auc, auc_times[1]-auc_times[0] )
            if (np.sum(spectra) > 0.0001):
                figOfScr, axOfScr = plt.subplots() 
                axOfScr.step( frq, spectra )
                axOfScr.set_title("Neuron spectra of " + numberOfBound) 
                figOfScr.savefig(neuronDir + "Neuron spectra_" + numberOfBound, dpi=500)
                meanFr = sp_in_bound.size/( bound["upper_bound"] - bound["low_bound"] )
            

            statisticsByNeuron += ("""
            %s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n
            """ % (bound["comment"], bound["low_bound"], bound["upper_bound"], sp_in_bound.size, meanFr, cv, tau_maxs, modeFr) )
                
            statisticsByAllNeurons += ("%d \t %f \t %f \t %f \t %f \t"  %  (sp_in_bound.size, meanFr, cv, tau_maxs, modeFr) )
        #else:
            #statisticsByAllNeurons += ("%d \t %f \t %f \t %f \t %f \t"  %  (sp_in_bound.size, 0, 0, 0, 0) )
        plt.show("all")
        plt.close("all")
        
        statisticsByAllNeurons += "\n"
        fileOfstat = open(neuronDir + "stat.txt", 'w')   
        fileOfstat.write(statisticsByNeuron)
        fileOfstat.close()
        
        #break
    #break

fileOfAllstat = open(saveDir + "Whole_stat.txt", 'w')   
fileOfAllstat.write(statisticsByAllNeurons)
fileOfAllstat.close()
