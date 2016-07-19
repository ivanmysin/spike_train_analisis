# -*- coding: utf-8 -*-
"""
concatenate wav-file stimulations to one file
"""

import os
import numpy as np
from scipy.io.wavfile import read, write
source_path = "/home/ivan/Data/Ach/origin_data/"
target_path = "/home/ivan/Data/Ach_full/source_data_simulations/tmp/"

counter = 1
for idx, folder in enumerate(sorted(os.listdir(source_path))):
    if (folder[0] == '.' or folder == 'not cutted'):
        continue
    
    print (folder)
    record_path = source_path + folder + '/'
    wavdata = np.empty( (0, 2), dtype=np.int16 )
    for wavfile in sorted(os.listdir(record_path)):
        name, ext = os.path.splitext(wavfile)
        if ( ext == ".wav" and name.isdigit() and len(name) > 3):
            print ("--->" + name)
            wavcontent = read(record_path + wavfile)
            fd = wavcontent[0]
            wavdata = np.append(wavdata, wavcontent[1], axis=0)
    if (wavdata.size > 2):
        write(target_path + folder + ".wav", fd, wavdata)

        
print (fd)

