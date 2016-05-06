# -*- coding: utf-8 -*-
"""
merge spike cluster
"""
import numpy as np
import scipy.io.matlab as matl
import matplotlib.pyplot as plt
source_path = "/home/ivan/Data/Ach_full/processing/discriminated_spikes/"
target_path = "/home/ivan/Data/Ach_full/processing/discriminated_spikes_merged/"

source_file = "211int_discr_channel_2.mat"

data = matl.loadmat(source_path + source_file)

new_data = {}

new_data["1"] = data["1"]
new_data["1"] = np.append(new_data["1"], data["3"])
#new_data["1"] = np.append(new_data["1"], data["4"])
#new_data["1"] = np.append(new_data["1"], data["5"])
#new_data["1"] = np.append(new_data["1"], data["8"])

new_data["2"] = data["2"]
new_data["2"] = np.append(new_data["2"], data["4"])
new_data["2"] = np.append(new_data["2"], data["5"])
#new_data["2"] = np.append(new_data["2"], data["14"])
#new_data["2"] = np.append(new_data["2"], data["15"])

#new_data["3"] = data["5"]
#new_data["3"] = np.append(new_data["3"], data["9"])
#new_data["4"] = data["9"]
#new_data["3"] = np.append(new_data["3"], data["15"])
#new_data["3"] = np.append(new_data["3"], data["16"])
#new_data["3"] = np.append(new_data["3"], data["24"])
#new_data["3"] = np.append(new_data["3"], data["26"])
#
#
#
#new_data["4"] = data["17"]
#new_data["4"] = np.append(new_data["4"], data["18"])
#new_data["4"] = np.append(new_data["4"], data["19"])
#new_data["4"] = np.append(new_data["4"], data["20"])
#new_data["4"] = np.append(new_data["4"], data["21"])
#new_data["4"] = np.append(new_data["4"], data["22"])



for key in new_data.keys():
    new_data[key] = np.sort(new_data[key])
matl.savemat(target_path + source_file, new_data)

#splt.plot( new_data["2"] )