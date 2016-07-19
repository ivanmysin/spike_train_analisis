# -*- coding: utf-8 -*-
"""
process spikes train with markers
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
from matplotlib.widgets import Button


#######################################################
class SaveBounds:
    def __init__(self, line1, line2, savingfile):
        self.bound1 = np.array([]) 
        self.bound2 = np.array([])
        self.line1 = line1
        self.line2 = line2
        self.savingfile = savingfile
    
    def add(self, event):
        self.bound1 = np.append(self.bound1, line1.get_xdata()[0]) 
        self.bound2 = np.append(self.bound2, line2.get_xdata()[0]) 

    def save(self, event):
        f = open(self.savingfile, "w")
        for ind in range(self.bound1.size):
            print ( self.bound1[ind], self.bound2[ind] )
            f.write("%f \t %f \n" % (self.bound1[ind], self.bound2[ind]))
        f.close()

#########################################################
class LineBuilder:
    def __init__(self, line, bound):
        self.line = line
        self.xs = list()
        self.ys = list(line.get_ydata())
        self.bound = bound
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        # print ( 'click', event.button )
        if event.inaxes!=self.line.axes: return
        if self.bound == "left_bound" and event.button != 1: return
        if self.bound == "right_bound" and event.button != 3: return
        self.xs = np.array([event.xdata, event.xdata])
        # self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

######################################################################
def get_markers_from_file(pathfile):
    markers_file = open(pathfile, "r")
    file_contetnt = markers_file.read()
    markers_file.close()
    markers = []
    for idx, line in enumerate( file_contetnt.split("\n")[1:] ):
        if (line == "" or line == "\n"):
            continue
        item, key = line.split("\t")
        markers.append( {"time": float(item), 'lable': key} )
    return markers

######################################################################
def get_rate_plot(spikes, step):
    
    time_bins = np.arange(0, np.max(spikes), step)
    rate, _ = np.histogram(spikes, time_bins)    
    rate = rate.astype(dtype=float)/step

    return time_bins[:-1], rate
    
#######################################################################
main_path = "/home/ivan/Data/Ach_full/"
marker_path = main_path + "markers/"
discrim_spikes_path = main_path + "final_disrimination/discriminated_spikes/"
bounds_path = main_path + "bounds/"

counter = 1
for matfile in sorted(os.listdir(discrim_spikes_path)):
    if (matfile[0] == "."):
        continue
    print (matfile)

    markers_file = matfile.split("int")[0] + ".txt"
    markers = get_markers_from_file(marker_path + markers_file)

    mat_content = loadmat(discrim_spikes_path + matfile)
    
    for key in sorted(mat_content.keys()):
        spike_train = mat_content[key]
        if type(spike_train) is np.ndarray:
            spike_train = spike_train.reshape(spike_train.size)
            times, spikes_rate = get_rate_plot(spike_train, 10)

            if (np.mean(spikes_rate) > 0.5):
                continue
            
            counter += 1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.step(times, spikes_rate)
            ax.set_xlabel("time, sec")
            ax.set_ylabel("rate, sp/sec")
            ax.set_ylim(0, 2.5*spikes_rate.max())
            
            for idx, mark in enumerate(markers):
                time = mark["time"]
                lable = mark["lable"]
                if (idx%2 != 0):
                    y_text = spikes_rate.max() * ( np.random.rand() + 1)
                else:
                    y_text = 1.2 * spikes_rate.max() * ( np.random.rand() + 1)

                ax.annotate( lable, xy=(time, spikes_rate[np.argmin( np.abs(times - time) ) ] ), 
                        xytext=(time,  y_text), arrowprops=dict(facecolor='black', shrink=0.01), 
                        horizontalalignment='center')
            #fig.set_size_inches(30, 5)
            ax.set_title( 'Интегралка of %s, № %s' % (matfile, key) )
            
            line1,  = ax.plot([0.5, 0.5], [0, 100], 'g')
            line2,  = ax.plot([1, 1], [0, 100], 'r')
            linebuilder1 = LineBuilder(line1, "left_bound")
            linebuilder2 = LineBuilder(line2, "right_bound")
            
            bounds_file_path = bounds_path + os.path.splitext(matfile)[0] 
            bounds_file_path += "_neuron_" + key + ".txt"       
            
            callback = SaveBounds(line1, line2, bounds_file_path)
            add = fig.add_axes([0.7, 0.05, 0.1, 0.075])
            save = fig.add_axes([0.81, 0.05, 0.1, 0.075])
            add_button = Button(add, 'Add')
            add_button.on_clicked(callback.add)
            
            save_button = Button(save, 'Save')
            save_button.on_clicked(callback.save)
            plt.show(block=True)
            #plt.close()
            
            
    
    #break
print (counter)

