# -*- coding: utf-8 -*-
"""
class discriminator
"""

import numpy as np
from scipy import io
import scipy.signal as sig
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import preprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mynet
import mylib as lib
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, DBSCAN, SpectralClustering
from itertools import cycle
from scipy import stats
from scipy.stats import variation as cv
from scipy import interpolate
import autoencoder as keras_autoencoder
import os

class Discriminator:
    def __init__ (self, origin_signal, fd, window_size=5, winstart=2):
        self.origin_signal = origin_signal
        self.spikes_signal = origin_signal
        self.fd = fd
        self.window_size =int( window_size*fd*0.001) # convert to number of samples
        self.winstart = winstart*fd*0.001

        self.spikes = {}
        self.spikes_indexes = np.array([])
        self.n_clusters = 1
        
    def clear_from_stims(self, stims_series):
        for stim in stims_series.values():
            if not(type(stim) is np.ndarray):
                continue

            for st in stim:
                starts_inds = int (st*self.fd - 0.005*self.fd) 
                ends_inds = int (st*self.fd + 0.005*self.fd)
                self.origin_signal[starts_inds:ends_inds] = 0
                
                starts_inds = ends_inds + 1
                ends_inds = starts_inds + int(0.1*fd )
                
                ma = lib.moving_average(self.origin_signal[starts_inds:ends_inds], int( 0.002*self.fd ), "exponential")
                self.origin_signal[starts_inds:ends_inds] -= ma
        
        std = np.std(self.origin_signal)
        self.origin_signal[(self.origin_signal > 6*std) & \
                           (self.origin_signal < -6*std)] = 0
    def get_argextremums(self):
        dif = np.diff(self.spikes_signal)
        dif[dif < 0] = -2
        dif[dif > 0] = 2
        dif[dif == 0] = 1
    	
        lm = np.diff(dif)
        ext_ind = np.argwhere(lm != 0)
        ext_ind += 1
        ext = np.zeros_like(dif)
        ext[ext_ind] = dif[ext_ind]
        lmax_ind = np.argwhere(ext < 0)
        lmin_ind = np.argwhere(ext > 0)
        return (lmax_ind, lmin_ind)

    def find_spikes_indexes_over_theshold(self, high_pass_bound=500, adiaphoriaFactor=0.005):
        #loc_max_ind, loc_min_ind = self.get_argextremums()
        
        
        #loc_max_ind = np.asarray (sig.argrelextrema(self.spikes_signal, np.greater, order=1) )
        #loc_max_vals = spikes_signal[loc_max_ind]
        
               
        #loc_min_ind = np.asarray (sig.argrelextrema(self.spikes_signal, np.less, order=1) )
        #loc_min_vals = spikes_signal[loc_min_ind]
        self.threshold = 4 * np.median( np.abs(self.spikes_signal[0:-1:10]) / 0.6745 )
        
        self.spikes_indexes = np.append( \
                      np.argwhere(self.spikes_signal >= self.threshold), \
                      np.argwhere(self.spikes_signal <= -self.threshold) )
        self.spikes_indexes = np.sort(self.spikes_indexes)
        self.spikes_indexes = self.spikes_indexes[ np.diff( np.append(0, self.spikes_indexes) ) > (self.fd * adiaphoriaFactor) ]
        
        self.n_clusters = 1
        self.labels = np.zeros(self.spikes_indexes.size)
                
        return self

    def compress_spikes_form(self, weightsFile, datasets_matfile, loadWeights=True, saveWeights=True, resamplingFactor=6, lr=0.1):
        if (resamplingFactor>1):
            t = np.linspace(0, self.window_size/self.fd, self.window_size)
            ts = np.linspace(0, self.window_size/self.fd, self.window_size*resamplingFactor)
        
        self.spikes_indexes = self.spikes_indexes[np.logical_and( (self.spikes_indexes > self.winstart), \
                (self.spikes_signal.size - self.spikes_indexes) > self.window_size)  ]
        
        if (self.spikes_indexes.size < 5):
            return False
        
        datasetOfSpikes = np.empty((self.spikes_indexes.size, self.window_size*resamplingFactor), dtype=float)
        
        for idx, sp_idx in enumerate(self.spikes_indexes):
            
        
            max_idx = np.argmax(self.spikes_signal[sp_idx-self.window_size:sp_idx+self.window_size] ) - self.window_size 
            l_ind = int(sp_idx - self.winstart + max_idx)
            h_ind = int(l_ind + self.window_size) 
                    
            if (h_ind > self.spikes_signal.size-1):
                continue
                    
            examp = self.spikes_signal[l_ind:h_ind]
            self.spikes_indexes[idx] += max_idx
            

            if (resamplingFactor > 1):
                f = interpolate.interp1d(t, examp, kind='cubic')
                examp = f(ts)
            
            datasetOfSpikes[idx, :] = examp
        
        datasetOfSpikes = datasetOfSpikes**3
        #preprocessing.normalize(datasetOfSpikes, copy=False)
        
        self.spikes = self.spikes_indexes / self.fd
        if (datasetOfSpikes.shape[0] < 5):
            return False
        #preprocessing.normalize(datasetOfSpikes, axis=1, copy=False)
        #datasetOfSpikes = np.abs( sig.hilbert(datasetOfSpikes) )

        compress_data = keras_autoencoder.compress_data_by_conv_autoencoder(datasetOfSpikes, 10, 40, weightsFile)
        io.savemat(datasets_matfile, {"dataset":datasetOfSpikes, 
                "compressed_dataset":compress_data, "spikes_times":self.spikes})
        

        compressor = PCA(n_components = 3)
        # manifold.Isomap(n_neighbors=30, n_components=3) 
        # 
        # 
        # manifold.TSNE(n_components=3, init='pca', random_state=0)
 
        self.spikes_compressed = compressor.fit_transform(compress_data)
        #np.empty((compress_data.shape[0], 3), dtype=float)
#        
#        min_idx = 0
#        max_idx = 5000
#        while(min_idx < compress_data.shape[0]):
#            compressed_slice = compress_data[min_idx:max_idx, :]
#            print (compressed_slice.shape)
#            self.spikes_compressed[min_idx:max_idx, :] = compressor.fit_transform(compressed_slice)
#            min_idx += 5000
#            max_idx += 5000

        return self

    def clusterize_spikes(self, n_clusters = None):
        
        if (n_clusters):
            clusterizator = KMeans(n_clusters=n_clusters)  
        else:
            bandwidth = estimate_bandwidth(self.spikes_compressed, quantile=0.8)
            clusterizator = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False) 
        #lib.XMeans() 

        # SpectralClustering(n_clusters=2)
#        DBSCAN(eps=0.05, min_samples=1000, \
#                              metric='euclidean', algorithm='auto', \
#                              leaf_size=70, p=None, random_state=None)
        
        
        clusterizator.fit(self.spikes_compressed)
        self.labels = clusterizator.labels_
        #cluster_centers = m_shift.cluster_centers_
        self.n_clusters = len(np.unique(self.labels))
   
        return self

    def get_spikes(self):
        spikes = {}
        if (self.n_clusters == 1):
            spikes[str(1)] = self.spikes_indexes/self.fd
        else:
            for idx in range (self.n_clusters):
                sp = self.spikes_indexes[self.labels==idx]
                spikes[str(idx+1)] = sp / self.fd
        return spikes

    def clear_outliers(self):
        new_compress_data = np.empty((0, self.spikes_compressed.shape[1]), dtype=np.float64)
        new_sp = np.empty((0, 1), dtype=int)
        for idx in range (self.n_clusters):
            this_cluster = np.copy ( self.spikes_compressed[self.labels == idx, :] )
            this_indexes = np.copy ( self.spikes_indexes[self.labels == idx] )

            if (this_indexes.size > 10):
                print ("N = " + str(this_indexes.size) )
                clear_cluster_ind = lib.clear_outliers(this_cluster)
                
                new_compress_data = np.append(new_compress_data, this_cluster[clear_cluster_ind], axis=0)
                new_sp = np.append(new_sp, this_indexes[clear_cluster_ind])
        
        if (new_sp.size < 20):
            print ('Очень мало осталось элементов в записи')
            return False
        else:
            self.spikes_compressed = new_compress_data
            self.spikes_indexes = new_sp
        return self
    
    def plot_signal(self, path, step_of_signal=10, start_window_of_signal=0, end_window_of_signal=5, max_time=-1):
        if (max_time == -1):
            max_time = self.origin_signal.size/self.fd

        times_samles = np.linspace(0, self.origin_signal.size/self.fd, self.origin_signal.size )
        signal_fig = plt.figure()
        signal_ax = signal_fig.add_subplot(111)
        signal_ax.plot(times_samles, self.origin_signal, "b")
        
        

        signal_ax.plot([0, times_samles[-1]], [self.threshold, self.threshold], "r", linewidth=2)
        signal_ax.set_ylim(-1.0, 1.0)
        while( end_window_of_signal < max_time):
            signal_ax.set_xlim(start_window_of_signal, end_window_of_signal)
            fileSignal = path + "signal_" + str(start_window_of_signal) + \
                                             "_" + str(end_window_of_signal) + ".png"

            signal_fig.savefig(fileSignal, dpi=200)
            start_window_of_signal += step_of_signal
            end_window_of_signal += step_of_signal

        return self
    


    def plot_results(self, plotTitle, spikeFormsFiles, pathSignals, pca_plot, step_of_signal=10, start_window_of_signal=0, end_window_of_signal=5, max_time=-1):
        import matplotlib.pyplot as plt        
        if (max_time == -1):
            max_time = self.origin_signal.size/self.fd
        times_samles = np.linspace(0, self.spikes_signal.size/self.fd, self.spikes_signal.size )
        signal_fig = plt.figure()
        signal_ax = signal_fig.add_subplot(111)
        signal_ax.plot(times_samles, self.spikes_signal, "b")
        signal_ax.set_ylim(-1.0, 1.0)
       
        signal_ax.plot([0, times_samles[-1]], [self.threshold, self.threshold], "r", linewidth=1)
        signal_ax.plot([0, times_samles[-1]], [-self.threshold, -self.threshold], "r", linewidth=1)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = cycle(["b", "g", "m", "c", "k", "y", "r"])
    
        for idx in range(self.n_clusters):
            sp = self.spikes_indexes[self.labels==idx]
            cluster_color = next(colors)
            if hasattr(self, 'spikes_compressed'):
                xs = self.spikes_compressed[self.labels==idx, 0]
                ys = self.spikes_compressed[self.labels==idx, 1]
                zs = self.spikes_compressed[self.labels==idx, 2]
                ax.scatter(xs, ys, zs, s=50, c=cluster_color)
                #ax.scatter(xs, ys, s=10, c=cluster_color)
                

            signal_ax.scatter(times_samles[sp], 0.7*np.ones(sp.size)+(idx*0.05),\
                                      c=cluster_color, s=50, marker="o")

            fig_of_spikes = plt.figure()
            ax_of_spikes = fig_of_spikes.add_subplot(111)
            
            
            for i in range (sp.size):
                l_ind = sp[i] - self.winstart
                h_ind = l_ind + self.window_size
                ax_of_spikes.plot(self.spikes_signal[l_ind:h_ind])
            ax_of_spikes.set_title( plotTitle + '(cluster №: ' + str(idx+1) + ')')
            ax_of_spikes.set_ylim(-1, 1)
            fig_of_spikes.savefig( spikeFormsFiles + 'cluster_' + str(idx+1) + ".png")
            
        if ( os.path.isdir(pathSignals) ):

            while( end_window_of_signal < max_time ):
                signal_ax.set_xlim(start_window_of_signal, end_window_of_signal)
                fileSignal = pathSignals + "signal_" + str(start_window_of_signal) + \
                                              "_" + str(end_window_of_signal) + ".png"

                signal_fig.savefig(fileSignal, dpi=200)
                start_window_of_signal += step_of_signal
                end_window_of_signal += step_of_signal



            ax.set_title(plotTitle + '(N of clusters: %d)' % self.n_clusters )
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_zlabel('PCA 3')
            
            fig.savefig(pca_plot, dpi=300)
            
            plt.show(block=False)
            plt.close(signal_fig)
            plt.show(block=False)
            plt.close(fig)
            plt.show(block=False)
            plt.close(fig_of_spikes)
            plt.close("all") #signal_fig, fig, fig_of_spikes
            
            return self
    
    def delete_spike_by_index(self, indexes):
        self.spikes_indexes = self.spikes_indexes[indexes]
        self.labels = self.labels[indexes]
        #self.spikes_compressed = self.spikes_compressed[:, indexes]
                
###############################################################################
def descriminate_stimulations(signal, fd, min_interstim_interval=0.25, threshold = 0.5):
    
    
    signal =  2 * ( signal - signal.min() ) / (signal.max() - signal.min()) - 1
    stims_find = np.ones(signal.shape[0], dtype=float)

    for ch_ind in range(signal.shape[1]):
        channel_signal = signal[:, ch_ind]
        
        stims_find *= np.abs(channel_signal)
        
    if (np.sum(stims_find) < 5):
        return []
    
    t = np.linspace(0, signal.shape[0]/fd, signal.shape[0])
    signal = np.abs(signal[:, 0]) * np.abs(signal[:, 1])
    stims = t[signal >= threshold]
        
    stims = stims[np.append([min_interstim_interval], np.diff(stims)) >= min_interstim_interval ]
    
    if (stims.size < 3):
        return []
    
    stimulations = {}
    stims_intervals = np.diff(stims)
    
    start_ind = 0
    end_ind = -1
    previous_interval = 2*stims_intervals[0]
    counter = 1
    for idx, st_int in enumerate(stims_intervals):
        if ( np.abs(st_int - previous_interval) / st_int  > 0.3 or  (idx == stims_intervals.size - 1)):
            if ( idx < stims_intervals.size - 1 ):
                end_ind = idx
            else:
                end_ind = idx + 1
            
            if ( (end_ind - start_ind) > 3 and cv(stims_intervals[start_ind:end_ind] < 0.4) ):
                st = stims[start_ind:end_ind+1]
                stimulations[str(counter)] = st
                counter += 1
                #print (len(stimulations), st[0], st[-1], np.mean(stims_intervals[start_ind:end_ind]))
            
            start_ind = idx + 1
        previous_interval = st_int
    return stimulations

           
###############################################################################
def autodescriminate(origin_spikes_signal, fd, stimulations, weightsFile, dataDescr, 
                     spikeFormsFiles, pca_plot, datasets_matfile,
                     window_size=3, winstart=1, saveWeights=True,
                     loadWeights=True, pathForSignal=False, 
                     resamplingFactor=6, adiaphoriaFactor=0.003, n_clusters = 0):
                         
    discrim = Discriminator(origin_spikes_signal, fd, window_size, winstart)
    if ( stimulations ):
        discrim.clear_from_stims(stimulations)
    discrim.find_spikes_indexes_over_theshold()
    
    spikes = discrim.get_spikes()["1"]
    needed_spikes = np.zeros_like(spikes, dtype=bool)
    for idx, st in stimulations.items():
        min_t = st[0] - 180
        max_t = st[-1] + 180
        needed_spikes = np.logical_or(needed_spikes, (spikes >= min_t) & (spikes <= max_t) )
    
    discrim.delete_spike_by_index(needed_spikes)   

    if ( n_clusters > 1):
        res = discrim.compress_spikes_form(weightsFile, datasets_matfile, loadWeights=True, saveWeights=True, resamplingFactor=resamplingFactor, lr=0.1)
        if (type(res) is bool):
            return False
        discrim.clusterize_spikes(n_clusters)
        #discrim.clear_outliers()
        #discrim.clusterize_spikes()
    discrim.plot_results(dataDescr, spikeFormsFiles, pathForSignal, pca_plot, \
                         step_of_signal=4, start_window_of_signal=0, \
                         end_window_of_signal=3, max_time=-1)
    spikes = discrim.get_spikes()

    return spikes
###############################################################################


if __name__ == "__main__":
    from scipy.io.wavfile import read
    main_path = '/home/ivan/Data/Ach_full/'
    
    descr_stims_path = main_path + 'discrimination_simulation/'

    
    dataDir = main_path + "source_data_simulations/"
    if not ( os.path.isdir(dataDir) ):
        raise SystemExit('Can not find source data directory!')
        
    processing_dir = main_path + 'processing_stimulations/'
    if not ( os.path.isdir(processing_dir) ):
        os.mkdir(processing_dir)
        
        
    discriminated_spikes_dir = processing_dir + "discriminated_spikes/"
    
    if not ( os.path.isdir(discriminated_spikes_dir) ):
        os.mkdir(discriminated_spikes_dir)
     

    stimulationDir = processing_dir + "discriminated_stimulations/"
    if not ( os.path.isdir(stimulationDir) ):
        os.mkdir(stimulationDir)    
    
    spikeFormsPath = processing_dir + "spikes_forms/"
    if not ( os.path.isdir(spikeFormsPath) ):
        os.mkdir(spikeFormsPath)
    
    pathSignals = processing_dir + "signals/"
    if not ( os.path.isdir(pathSignals) ):
        os.mkdir(pathSignals)
    
    
    
    
    ######################################
    loadWeights = True
    saveWeights = False
    weightsFile = main_path + "weights/conv_weigths_difficult**3.hdf5" # None
    
    
    for dataFile in sorted( os.listdir(dataDir) ):
        #dataFile = dataFile2["file"]
        if (os.path.splitext(dataFile)[1] != '.wav'): continue
            
        #if not("02-17-2016_1_CL_пикротоксин_стимуляция_скорость.wav" in dataFile):
        #    continue
        print (dataFile)
        
        data = read(dataDir + dataFile) # io.loadmat(dataDir + dataFile)
        fd = float(data[0])
        resamplingFactor = 1 #16000.0/fd
        ######################################

        spikes_signals = data[1].astype(dtype=float)
        if (len(spikes_signals.shape) == 1):
            spikes_signals = spikes_signals.reshape(spikes_signals.size, 1)
        #spikes_signals = spikes_signals[120000:840000, :] # !!!!!!!!!
        
        
        stims_file = descr_stims_path + os.path.splitext(dataFile)[0] + '_stims_descr.mat'
                
        if (os.path.isfile(stims_file)):
            stimulations = io.loadmat(stims_file)
            keys = list (stimulations.keys())
            for key in keys:
                stims = stimulations[key]
                if (type(stims) is np.ndarray):
                    stimulations[key] = stims.reshape(stims.size)
                else:
                    stimulations.pop(key)
        else:
            print ("Файл со стимуляциями не обнаружили и дискриминируем стимуляции")
            stimulations = descriminate_stimulations(spikes_signals, fd)
        
        
        for ch_ind in range(spikes_signals.shape[1]):
            #ch_ind = int(ch_ind-1)
            dataFile = os.path.splitext(dataFile)[0]
            resultFile = discriminated_spikes_dir  + dataFile + '_discr_channel_'  + str(ch_ind+1)
            
            
            spikes_signal = spikes_signals[:, ch_ind]
            spikes_signal = 2*( spikes_signal - spikes_signal.min() ) / (spikes_signal.max() - spikes_signal.min() ) - 1
            
            if ( os.path.isfile(resultFile + ".mat") ):
                continue

            pathForSignal = pathSignals + dataFile + "_channel_" + str(ch_ind+1) + "/"

           
            if not ( os.path.isdir(pathForSignal) ):
                os.mkdir(pathForSignal)

            spikes_forms = spikeFormsPath + '/' + dataFile + '_channel_' + str(ch_ind+1) + '/'
            
            if not ( os.path.isdir(spikes_forms) ):
                os.mkdir(spikes_forms)
            
            spikeFormsFiles =  spikes_forms 
            pca_plot = spikes_forms + dataFile + '_pca_channel_' + str(ch_ind+1) + ".png"

            datasets_matfile = spikes_forms + dataFile + "_datasets_" + str(ch_ind+1) + ".mat"
            
            spikes_signal = lib.butter_bandpass_filter(spikes_signal, 500, fd/2-100,fd)
            
            
            n_clusters_file = main_path + 'old_processing_stimulations/discriminated_spikes/' + dataFile + '_discr_channel_'  + str(ch_ind+1) + '.mat'
            if os.path.isfile(n_clusters_file):
                descrim = io.loadmat(n_clusters_file)
                
                if ("2" in descrim.keys()):
                    n_clusters = 2
                else:
                    n_clusters = 1
    
            else:
                n_clusters = 0
            descrim = {}


            
            
            
            spikes = autodescriminate(spikes_signal, fd, stimulations, weightsFile, \
                                        dataFile, spikeFormsFiles, pca_plot, datasets_matfile, 5, 2, saveWeights, \
                                      loadWeights, pathForSignal, resamplingFactor, 0.003, n_clusters)

            if(type(spikes) is bool):
                continue
            #resultFile = discriminated_spikes_dir  + dataFile + '_discr_channel_'  + str(ch_ind+1)

            io.savemat(resultFile, spikes)
            del(spikes)
            resultFile = stimulationDir + dataFile  + '_discr_stimulations'
            if (stimulations):
                io.savemat(resultFile, stimulations)
    