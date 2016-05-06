# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
from scipy import io
import os
loadpath = "/home/ivan/Документы/programming/python/processor/results3/"
savepath = "/home/ivan/Документы/programming/python/processor/discrimitation_results_in_txt/"
for matfile in os.listdir(loadpath):
    print (matfile)
    spikes = io.loadmat(loadpath + matfile)
    matfile = os.path.splitext(matfile)[0]

    matfile = matfile.split("discr")[1]
    
    for key in spikes.keys():
        if not (type(spikes[key]) is np.ndarray): continue
        sp = spikes[key][0]
        if (sp.size < 20): continue
            
        txtFile = open(savepath + "neuron_" + key + matfile + ".txt", "w", newline="\n") 
        #txtFile.write(key + "\n")
        for s in sp:
            txtFile.write( str(s) + "\n" )
        
        txtFile.close()
