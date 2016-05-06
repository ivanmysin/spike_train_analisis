# -*- coding: utf-8 -*-
"""
class discriminator
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
from scipy import stats
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
        self.threshold = 4*np.median( np.abs(self.spikes_signal[0:-1:10]) / 0.6745 )
        
        self.spikes_indexes = np.append( \
                      np.argwhere(self.spikes_signal >= self.threshold), \
                      np.argwhere(self.spikes_signal <= -self.threshold) )
        self.spikes_indexes = np.sort(self.spikes_indexes)
        self.spikes_indexes = self.spikes_indexes[ np.diff( np.append(0, self.spikes_indexes) ) > (self.fd * adiaphoriaFactor) ]
        
                
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
            #datasetOfSpikes = np.append(datasetOfSpikes, examp.reshape(1, self.window_size*resamplingFactor), axis=0)
        
        self.spikes = self.spikes_indexes / self.fd
        if (datasetOfSpikes.shape[0] < 5):
            return False
        #preprocessing.normalize(datasetOfSpikes, axis=1, copy=False)
        #datasetOfSpikes = np.abs( sig.hilbert(datasetOfSpikes) )

        compress_data = keras_autoencoder.compress_data_by_conv_autoencoder(datasetOfSpikes, 10, 40, weightsFile)
        io.savemat(datasets_matfile, {"dataset":datasetOfSpikes, 
                "compressed_dataset":compress_data, "spikes_times":self.spikes})
        
        pca = PCA(n_components=3)
        self.spikes_compressed = pca.fit(compress_data).transform(compress_data)
        return self

    def clusterize_spikes(self):
        bandwidth = estimate_bandwidth(self.spikes_compressed)

        m_shift = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=-1)
        m_shift.fit(self.spikes_compressed)
        self.labels = m_shift.labels_
        #cluster_centers = m_shift.cluster_centers_
        self.n_clusters = m_shift.cluster_centers_.shape[0]
   
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
    
        for idx in range (self.n_clusters):
            sp = self.spikes_indexes[self.labels==idx]
            xs = self.spikes_compressed[self.labels==idx, 0]
            ys = self.spikes_compressed[self.labels==idx, 1]
            zs = self.spikes_compressed[self.labels==idx, 2]
            cluster_color = next(colors)
            ax.scatter(xs, ys, zs, s=50, c=cluster_color)

            signal_ax.scatter(times_samles[sp], 0.8*np.ones(sp.size),\
                                      c=cluster_color, s=100, marker="o")

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
###############################################################################
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

           
###############################################################################
def autodescriminate(origin_spikes_signal, fd, stimulations, weightsFile, dataDescr, 
                     spikeFormsFiles, pca_plot, datasets_matfile,
                     window_size=3, winstart=1, saveWeights=True,
                     loadWeights=True, pathForSignal=False, resamplingFactor=6, adiaphoriaFactor=0.003 ):
                         
    discrim = Discriminator(origin_spikes_signal, fd, window_size, winstart)
    if ( stimulations ):
        discrim.clear_from_stims(stimulations)
    discrim.find_spikes_indexes_over_theshold()
    
    
    res = discrim.compress_spikes_form(weightsFile, datasets_matfile, loadWeights=True, saveWeights=True, resamplingFactor=resamplingFactor, lr=0.1)
    if (type(res) is bool):
        return False
    discrim.clusterize_spikes()
    #discrim.clear_outliers()
    #discrim.clusterize_spikes()
    discrim.plot_results(dataDescr, spikeFormsFiles, pathForSignal, pca_plot, \
                         step_of_signal=10, start_window_of_signal=0, \
                         end_window_of_signal=5, max_time=-1)
    spikes = discrim.get_spikes()

    return spikes
###############################################################################


if __name__ == "__main__":
    from scipy.io.wavfile import read
    main_path = '/home/ivan/Data/Ach_full/'
    #'/home/ivan/Data/speed_of_streem/test_7_layers/!Без пикротоксина/'
    # '/home/ivan/Data/Ach_full/'
    #'/home/ivan/Data/speed_of_streem/!Без пикротоксина/'
    #'/home/ivan/Data/test_of_spike_sorting/!Без пикротоксина/'
    #'/home/ivan/Data/speed_of_streem/Свинки/2_Цельные записи после очистки/'

    
    dataDir = main_path + "source_data/"
    if not ( os.path.isdir(dataDir) ):
        raise SystemExit('Can not find source data directory!')
        
    processing_dir = main_path + 'processing/'
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
    
    
    pca_plots = processing_dir + "pca_plots/"
    if not ( os.path.isdir(pca_plots) ):
        os.mkdir(pca_plots)
    
    
    ######################################
    loadWeights = True
    saveWeights = True
    weightsFile = main_path + "weights/conv_weigths.hdf5" # None

    for dataFile in os.listdir(dataDir):

        if (os.path.splitext(dataFile)[1] != '.wav'): continue
            
        #if not("02-17-2016_1_CL_пикротоксин_стимуляция_скорость.wav" in dataFile):
        #    continue
            
        print (dataFile)


        data = read(dataDir + dataFile) # io.loadmat(dataDir + dataFile)
        fd = float(data[0])
        resamplingFactor = 1 #16000.0/fd
        ######################################

        spikes_signals = data[1].astype(dtype=float)
        #spikes_signals = spikes_signals[120000:840000, :] # !!!!!!!!!
        
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        stimulations = False # descriminate_stimulations(spikes_signals, fd)
        for ch_ind in range(spikes_signals.shape[1]):
            dataFile = os.path.splitext(dataFile)[0]
            resultFile = discriminated_spikes_dir  + dataFile + '_discr_channel_'  + str(ch_ind+1)
            
            print (resultFile)
            if ( os.path.isfile(resultFile + ".mat") ):
                continue
                
            spikes_signal = spikes_signals[:, ch_ind]
            spikes_signal = 2*( spikes_signal - spikes_signal.min() ) / (spikes_signal.max() - spikes_signal.min() ) - 1


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
            spikes = autodescriminate(spikes_signal, fd, stimulations, weightsFile, \
                                        dataFile, spikeFormsFiles, pca_plot, datasets_matfile, 5, 2, saveWeights, \
                                      loadWeights, pathForSignal, resamplingFactor)

            if(type(spikes) is bool):
                continue
            #resultFile = discriminated_spikes_dir  + dataFile + '_discr_channel_'  + str(ch_ind+1)

            io.savemat(resultFile, spikes)
            del(spikes)
        resultFile = stimulationDir + dataFile  + '_discr_stimulations'
        if (stimulations):
            io.savemat(resultFile, stimulations)
        #break
    #break




