# -*- coding: utf-8 -*-
"""
spike sorter sctipt
"""

import numpy as np
from scipy import io
import scipy.signal as sig
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mynet
import mylib as lib
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

from scipy.io.wavfile import read
from scipy import interpolate
import os
############################################################################



############################################################################
def autodescriminate(origin_spikes_signal, fd, weightsFile, dataDescr, spikeFormsFiles,
                     window_size=5, winstart=2, saveWeights=True, 
                     loadWeights=True, pathForSignal=False, resamplingFactor=6, adiaphoriaFactor=0.003 ):

    if not (type(weightsFile) is str):
        loadWeights = False
        saveWeights = False
    
    window_size = window_size*fd*0.001 # take 3 ms
    winstart = winstart*fd*0.001 # 1 ms from start
    # winend = window_size - winstart
    lr = 0.1
    #####################################
    
    order = int(fd * adiaphoriaFactor)
    spikes_signal = lib.butter_bandpass_filter(origin_spikes_signal, 500, fd/2, fd)
    loc_max_ind = np.asarray ( sig.argrelextrema(spikes_signal, np.greater, order=order) )
    loc_max_vals = spikes_signal[loc_max_ind]
    threshold = 4*np.median( np.abs(spikes_signal) / 0.6745 ) #np.percentile(loc_max_vals, 99)  #np.sqrt(np.std(spikes_signal)) #2*
    
    spikes_indexes = loc_max_ind[ loc_max_vals >= threshold ]
    
    print ("N of loc max %d and > of threshold %d" % (loc_max_vals.size, spikes_indexes.size) )

    


    datasetOfSpikes = np.empty((0, window_size*resamplingFactor), dtype=float) 
    if (resamplingFactor>1):
        t = np.linspace(0, window_size/fd, window_size)
        ts = np.linspace(0, window_size/fd, window_size*resamplingFactor)
    for i in range (spikes_indexes.size):
        l_ind = spikes_indexes[i] - winstart
        h_ind = l_ind + window_size
        if ( (l_ind < 0) or (h_ind>spikes_signal.size) ):
            continue
        examp = np.copy(spikes_signal[l_ind:h_ind])  #lib.preprocess( signal_spikes[l_ind:h_ind] )
        #examp = examp.reshape(1, window_size)
        if (resamplingFactor > 1):
            f = interpolate.interp1d(t, examp, kind='cubic')
            examp = f(ts)

        datasetOfSpikes = np.append(datasetOfSpikes, examp.reshape(1, window_size*resamplingFactor), axis=0)
    
    preprocessing.normalize(datasetOfSpikes, axis=0, copy=False)   
    
    # declare network
    dataset = { "input": datasetOfSpikes, "target": datasetOfSpikes }
    layers = [
        {"n": window_size*resamplingFactor, "bias": True, },
        {"n": window_size*resamplingFactor, "bias": True, },
        {"n": 10, "bias": True, },
        {"n": window_size*resamplingFactor, "bias": True,},
        {"n": window_size*resamplingFactor, "bias": True,}
    ]
    
    autoencoder = mynet.Network(layers, dataset["input"].shape[1])  
    if (loadWeights):
        weightsData = io.loadmat(weightsFile)
        weights = []
        for idx in range(len(layers)):
            weights.append(weightsData[str(idx)])
        autoencoder.setWeights(weights)
    
    save_err = autoencoder.trainOnDataset(dataset, 0.1, 100, lr, 50)
    print (save_err)
    weights = autoencoder.getWeights() 
    if (saveWeights):
        weightsData = {"data":{}}
        for idx, l in enumerate(weights):
            weightsData["data"][str(idx)] = l
        io.savemat(weightsFile, weightsData["data"])
    
    coder = mynet.Network(layers[0:3], dataset["input"].shape[1])
    coder.setWeights(weights[0:3])
    compress_data = coder.activate(dataset["input"])
    pca = PCA(n_components=3)
    
    spikes_pca = pca.fit(compress_data).transform(compress_data)
    
    ###############################################################################
    # Compute clustering with MeanShift
    # The following bandwidth can be automatically detected using
    # print (spikes_pca.shape)
    bandwidth = estimate_bandwidth(spikes_pca)
    
    m_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    m_shift.fit(spikes_pca)
    labels = m_shift.labels_
    cluster_centers = m_shift.cluster_centers_
    n_clusters_ =  cluster_centers.shape[0]
    
    
    new_compress_data = np.empty((0, spikes_pca.shape[1]), dtype=np.float64)
    new_sp = np.empty((0, 1), dtype=int)
    for idx in range (n_clusters_):
        this_cluster = np.copy ( spikes_pca[m_shift.labels_ == idx, :] )
        this_indexes = np.copy ( spikes_indexes[m_shift.labels_ == idx] )
        
        this_itervals = (this_indexes[1:-1] - this_indexes[0:-2])/fd
        good_intervals = this_itervals > adiaphoriaFactor
        this_cluster = this_cluster[good_intervals]
        this_indexes = this_indexes[good_intervals]
        
        if (this_indexes.size > 200):
            print ("N = " + str(this_indexes.size) )
            clear_cluster_ind = lib.clear_outliers(this_cluster)
            new_compress_data = np.append(new_compress_data, this_cluster[clear_cluster_ind], axis=0)
            new_sp = np.append(new_sp, this_indexes[ clear_cluster_ind ])
    if (new_sp.size < 20):
        print ('Очень мало осталось элементов в записи')
        return False
    bandwidth = estimate_bandwidth(new_compress_data)
    
    m_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    m_shift.fit(new_compress_data)
    labels = m_shift.labels_
    cluster_centers = m_shift.cluster_centers_
    n_clusters_ = cluster_centers.shape[0]

    #if (n_clusters_ > 10):
    #    print ("Очень много кластеров, n=" + str(n_clusters_) )
    #    return False

    ###############################################################################
    # form result of script
    
    times_samles = np.linspace(0, spikes_signal.size/fd, spikes_signal.size )
    signal_fig = plt.figure()
    signal_ax = signal_fig.add_subplot(111)    
    signal_ax.plot(times_samles, spikes_signal, "b")
    
    signal_ax.plot([0, times_samles[-1]], [threshold, threshold], "r", linewidth=2)    
    #signal_ax.set_xlim(0, 60)
    signal_ax.set_ylim(-1.0, 1.0)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    spikes = {}
    colors = cycle(["b", "g", "m", "c", "k", "y", "r"])
    for idx in range (n_clusters_):
        sp = new_sp[labels==idx] 
        spikes[str(idx+1)] = sp / fd

        xs = new_compress_data[labels==idx, 0]
        ys = new_compress_data[labels==idx, 1]
        zs = new_compress_data[labels==idx, 2]
        cluster_color = next(colors)
        ax.scatter(xs, ys, zs, s=50, c=cluster_color)
        signal_ax.scatter(times_samles[sp], spikes_signal[sp], c=cluster_color, s=30 )
        
        fig_of_spikes = plt.figure()
        ax_of_spikes = fig_of_spikes.add_subplot(111)
        
        for i in range (sp.size):
            l_ind = sp[i] - winstart
            h_ind = l_ind + window_size
            ax_of_spikes.plot(spikes_signal[l_ind:h_ind])
        
        ax_of_spikes.set_title(dataDescr + '(cluster №: ' + str(idx+1) + ')')
        fig_of_spikes.savefig( spikeFormsFiles + 'cluster_' + str(idx+1) )
        
    # save signal figure with different xlims
    if (os.path.isdir(pathSignals) ):

        step_of_signal = 10 # in second
        start_window_of_signal = 0
        end_window_of_signal = 5
        while( end_window_of_signal < 800): #times_samles[-1] # !!!!!!!!!!!!!!!!!!!!!
            signal_ax.set_xlim(start_window_of_signal, end_window_of_signal)
            fileSignal = pathForSignal + "signal_" + str(start_window_of_signal) + \
                                      "_" + str(end_window_of_signal) + ".png"
        
            signal_fig.savefig(fileSignal)
            start_window_of_signal += step_of_signal
            end_window_of_signal += step_of_signal
    
    
        
    ax.set_title(dataDescr + '(N of clusters: %d)' % n_clusters_ )
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.show()
    
    return spikes
######################################



dataDir =  "./data/" # 
# dataFile = 'Изм_протока_1_19-01-16.wav'      # "C_Easy1_noise01_short.mat"
resultDir = "./results5/"
spikeFormsPath = "./spikeFromsFigure4/"
pathSignals = "./signals4/"
######################################
loadWeights = False
saveWeights = True
weightsFile = "./weights/24Hz(5ms).mat"
resamplingFactor = 1.5
for dataFile in os.listdir(dataDir):
    #if not("(reg)" in dataFile): continue

    if (os.path.splitext(dataFile)[1] != '.wav'): continue 
    
    data = read(dataDir + dataFile) # io.loadmat(dataDir + dataFile)
    fd = float(data[0])
    # spikes_signal = data["data"][0]
    resamplingFactor = 24000.0/fd
    ######################################
    
    spikes_signals = data[1].astype(dtype=float)
    for ch_ind in range(spikes_signals.shape[1]):
        #if (ch_ind != 1): continue
        
        spikes_signal = spikes_signals[int(fd*6):-1, ch_ind]
        spikes_signal = 2*( spikes_signal - spikes_signal.min() ) / (spikes_signal.max() - spikes_signal.min() ) - 1
        """
        spikes_signal[spikes_signal > 7*np.std(spikes_signal) ] = 0 #np.percentile(spikes_signal, 99.99)
        spikes_signal[spikes_signal < -7*np.std(spikes_signal) ] = 0 # np.percentile(spikes_signal, 0.01)
        spikes_signal = 2*( spikes_signal - spikes_signal.min() ) / (spikes_signal.max() - spikes_signal.min() ) - 1
        """
        

        pathForSignal = pathSignals + os.path.splitext(dataFile)[0] + "_channel_" + str(ch_ind+1) + "/"
     
        if not ( os.path.isdir(pathForSignal) ):
            os.mkdir(pathForSignal)   
            
        dataFile = os.path.splitext(dataFile)[0]
        spikeFormsFiles = spikeFormsPath + dataFile + '_channel_' + str(ch_ind+1) + '_' 
        spikes = autodescriminate(spikes_signal, fd, weightsFile, \
                                    dataFile, spikeFormsFiles, 5, 2, saveWeights, \
                                  loadWeights, pathForSignal, resamplingFactor)
    
        if(type(spikes) is bool):
            continue
        resultFile = resultDir + 'discr_channel_'  + str(ch_ind+1) + '_' + dataFile
        io.savemat(resultFile, spikes)
        #break
    #break
