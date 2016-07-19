# -*- coding: utf-8 -*-
"""
processing of final discrimination with bounds
"""
import os
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyexcel
import pyexcel.ext.xlsx
import spike_train_lib as stl
from scipy.stats import mannwhitneyu

def parse_bounds(file_path):
    if not (os.path.isfile(file_path)):
        print (file_path + " File not found!!!")
        return {"bounds" : []}
    sheet = pyexcel.get_sheet(file_name=file_path)
    bound_dict = {
        "record"  : sheet.column[1][0],
        "channel" : sheet.column[1][1],
        "neuron"  : sheet.column[1][2],
        "bounds"  : [],    
    }
    for idx in range(len(sheet.column[0][3:])):
        idx += 3
        
        tmp_bounds = sheet.column[1][idx].split(" ")
        if (len(tmp_bounds) == 1 and tmp_bounds[0] == ""):
            continue

        if (len(tmp_bounds) != 2):
            tmp_bounds = []
            tmp_bounds.append( sheet.column[1][idx] )
            tmp_bounds.append( sheet.column[2][idx] )
            
        #tmp_bounds[0] = min(tmp_bounds)
        #tmp_bounds[1] = max(tmp_bounds)
            
        tmp_dict = {
            'name' : sheet.column[0][idx],
            'lower_bound' : float(tmp_bounds[0]),
            'upper_bound' : float(tmp_bounds[1]),        
        }
        
        bound_dict["bounds"].append(tmp_dict)
    return (bound_dict)
#############################################################
def save_as_xlsx(whole_stat_arr, xlsx_file):
    n_rows = len(whole_stat_arr)
    labels = []
    for idx, row in enumerate(whole_stat_arr):
        for key in row["stat"].keys():
            labels.append(key)
    
    uniq_params = list(row["stat"][key].keys()) 
    
    labels = sorted (set(labels))
    
    n_effects = len(uniq_params)

    n_cols = len(labels)*n_effects
    
    #формируем заколовки таблицы
    # массив эффектов
    effects = []
    for _ in range(n_effects):
        for lab in labels:
            effects.append(lab)
    effects = sorted(effects)
    # массив параметров обработки
    params = []
    for _ in range(len(labels)):
        for p in uniq_params:
            params.append(p)
    

        
    final_array = [[""], ["Neuron"]]
    for eff in effects:
        final_array[0].append(eff)
    for p in params:
        final_array[1].append(p) 

    for row_idx in range(n_rows):
        final_array.append([whole_stat_arr[row_idx]["Neuron"]])
        for col_idx in range(n_cols):
            param_key = params[col_idx]
            effect_key = effects[col_idx]
            if (effect_key in whole_stat_arr[row_idx]["stat"].keys()):
                value = whole_stat_arr[row_idx]["stat"][effect_key][param_key]
            else:
                value = "-"
            final_array[-1].append(value)
            
    whole_stat = pyexcel.Sheet(final_array)
    whole_stat.save_as(xlsx_file)
    return True

#####################################################################
main_path = '/home/ivan/Data/Ach_full/'
discr_path = main_path + 'final_disrimination/discriminated_spikes/'
bounds_path = main_path + 'bounds/'
result_path = main_path + 'statistics/'
whole_stat_arr = []
effects_significance = [
    ["Neuron", "GABA (U value)", "GABA (p value)", "GABA (significance)", "N1", "N2", 
               "Pc + Phac (U value)", "Pc + Phac (p value)", "Pc + Phac (significance)", "N1", "N2",
               "Ezr (U value)", "Ezr (p value)", "Ezr (significance)", "N1", "N2",
               "Sc + Hex (U value)", "Sc + Hex (p value)", "Sc + Hex (significance)", "N1", "N2"
   ]
  ]
for matfile in sorted(os.listdir(discr_path)):
    if (matfile[0] == '.' or os.path.splitext(matfile)[1] != '.mat'):
        continue
    print (matfile)
    matcontent = loadmat(discr_path + matfile)
   
    
    for nn, spike_train in sorted(matcontent.items()):
        if not (type (spike_train) is np.ndarray):
            continue
        spike_train = spike_train.reshape(spike_train.size)
        if (int(nn) > 2):
        
            continue
        
        bounds_file = matfile[0:3] + '-' + matfile[-5] + '-' + nn + '.xlsx'
        print (bounds_file)
        bounds = parse_bounds(bounds_path + bounds_file) 
        neuron_name = matfile.split("_discr")[0] + "_channel_" + matfile[-5] + '_neuron_' + nn
        
#        effects_significance.append([neuron_name])
#        
#        
#        if ( len(bounds["bounds"]) >= 5 ):
#            
#            gaba_controle = spike_train[(spike_train <= bounds["bounds"][2]["upper_bound"]) & \
#                                        (spike_train >= bounds["bounds"][2]["lower_bound"])]
#            gaba_effect   = spike_train[(spike_train <= bounds["bounds"][3]["upper_bound"]) & \
#                                        (spike_train >= bounds["bounds"][3]["lower_bound"])]
#
#            
#            u_value, p_value = mannwhitneyu(np.diff(gaba_controle), np.diff(gaba_effect),  use_continuity=True, alternative='two-sided')            
#            if (p_value <= 0.05):
#                signific = 'yes'
#            else:
#                signific = 'no'
#            
#            effects_significance[-1].append(u_value)
#            effects_significance[-1].append(p_value)
#            effects_significance[-1].append(signific)
#            effects_significance[-1].append(gaba_controle.size-1)
#            effects_significance[-1].append(gaba_effect.size-1)
#        
#            
#            pc_phac_controle = spike_train[(spike_train <= bounds["bounds"][4]["upper_bound"]) & \
#                                           (spike_train >= bounds["bounds"][4]["lower_bound"])]
#            pc_phac__effect  = spike_train[(spike_train <= bounds["bounds"][5]["upper_bound"]) & \
#                                           (spike_train >= bounds["bounds"][5]["lower_bound"])]
#                                           
#            u_value, p_value = mannwhitneyu(np.diff(pc_phac_controle), np.diff(pc_phac__effect),  use_continuity=True, alternative='two-sided')
#            
#            if (p_value <= 0.05):
#                signific = 'yes'
#            else:
#                signific = 'no'
#            
#            effects_significance[-1].append(u_value)
#            effects_significance[-1].append(p_value)
#            effects_significance[-1].append(signific)
#            effects_significance[-1].append(pc_phac_controle.size-1)
#            effects_significance[-1].append(pc_phac__effect.size-1)
#        else:
#            for _ in range(10):
#                effects_significance[-1].append("-")
#        
#        if ( len(bounds["bounds"]) >= 11 ):
#            ezr_controle = spike_train[(spike_train <= bounds["bounds"][8]["upper_bound"]) & \
#                                        (spike_train >= bounds["bounds"][8]["lower_bound"])]
#            ezr_effect   = spike_train[(spike_train <= bounds["bounds"][9]["upper_bound"]) & \
#                                        (spike_train >= bounds["bounds"][9]["lower_bound"])]
#
#            u_value, p_value = mannwhitneyu(np.diff(ezr_controle), np.diff(ezr_effect),  use_continuity=True, alternative='two-sided')
#            
#            if (p_value <= 0.05):
#                signific = 'yes'
#            else:
#                signific = 'no'
#            
#            effects_significance[-1].append(u_value)
#            effects_significance[-1].append(p_value)
#            effects_significance[-1].append(signific)
#            effects_significance[-1].append(ezr_controle.size-1)
#            effects_significance[-1].append(ezr_effect.size-1)
#            
#            sc_hex_controle = spike_train[(spike_train <= bounds["bounds"][10]["upper_bound"]) & \
#                                           (spike_train >= bounds["bounds"][10]["lower_bound"])]
#            sc_hex_effect   = spike_train[(spike_train <= bounds["bounds"][11]["upper_bound"]) & \
#                                           (spike_train >= bounds["bounds"][11]["lower_bound"])]
#        
#            u_value, p_value = mannwhitneyu(np.diff(sc_hex_controle), np.diff(sc_hex_effect),  use_continuity=True, alternative='two-sided')
#            if (p_value <= 0.05):
#                signific = 'yes'
#            else:
#                signific = 'no'
#            
#            effects_significance[-1].append(u_value)
#            effects_significance[-1].append(p_value)
#            effects_significance[-1].append(signific)
#            effects_significance[-1].append(sc_hex_controle.size-1)
#            effects_significance[-1].append(sc_hex_effect.size-1)
#            
#        else:
#            for _ in range(10):
#                effects_significance[-1].append("-")



        
        
        neuron_dir = result_path + neuron_name
        neuron_dir += '/'
        
        whole_stat_arr.append({"Neuron": neuron_name, "stat":{} })
        if not (os.path.isdir(neuron_dir)):
            os.mkdir(neuron_dir)
            
        time_bins, spike_rate = stl.get_rate_plot(spike_train, 10)
        time_bins = time_bins[0:-1]
        fig_of_rate = plt.figure()
        ax_of_rate = fig_of_rate.add_subplot(111)
        ax_of_rate.step(time_bins, spike_rate)
        
        for bd in bounds["bounds"]:
            upper_bound = bd["upper_bound"]
            lower_bound = bd["lower_bound"]
            effect_name = bd["name"]
            print (lower_bound, upper_bound)
            if (upper_bound - lower_bound == 0):
                whole_stat_arr[-1]["stat"][effect_name] = {
                    "N of spikes": '-',
                    "Mean frequency": '-',
                    "CV": '-',
                    "Tau by maximums": '-',
                    "Mode frequency": '-',
                    }
                continue
            sp = spike_train[ (spike_train >= lower_bound) & (spike_train <= upper_bound) ]
            ax_of_rate.add_patch( patches.Rectangle((lower_bound, 0), (upper_bound - lower_bound), 150, alpha=0.1))
            if (sp.size > 2):            
                ax_of_rate.text(lower_bound - 2*len(effect_name), 1 / np.max(np.diff(sp) + 0.001) + 5,  effect_name)            
            else:
                ax_of_rate.text(lower_bound - 2*len(effect_name), 1,  effect_name)
            """
            if (sp.size == 0):
                whole_stat_arr[-1]["stat"][effect_name] = {
                    "N of spikes": 0,
                    "Mean frequency": 0,
                    "CV": 0,
                    "Tau by maximums": 0,
                    "Mode frequency": 0,
                }
                continue
            
            
            hmsi_bins, hmsi, cv = stl.get_hmsi(sp, 0.001, neuron_dir + 'hmsi_of_' + effect_name +".png", effect_name)
        
            auc_times, auc, tau_mins, tau_maxs  = stl.get_autororrelogram(sp, 0.001)
            figOfAuc, axOfAuc = plt.subplots() 
                    
            if ( np.sum(auc) > 0.001):
                axOfAuc.step(auc_times[0:-1], auc)
                axOfAuc.set_title("Autocorrelelogram of " + effect_name)      
                axOfAuc.set_ylim(0, 1.2*np.max(auc) )
                figOfAuc.savefig(neuron_dir + "Autocorrelelogram_" + effect_name, dpi=500)
                plt.show(block=False)
                
            frq, spectra, modeFr = stl.get_neuron_spectra(auc, auc_times[1]-auc_times[0] )
            
            if (np.sum(spectra) > 0.0001):
                figOfScr, axOfScr = plt.subplots() 
                axOfScr.step( frq, spectra )
                axOfScr.set_title("Neuron spectra of " + effect_name) 
            figOfScr.savefig(neuron_dir + "Neuron spectra_" + effect_name, dpi=500)
            meanFr = sp.size/( upper_bound - lower_bound )
            plt.show(block=False)
            plt.close("all")
            whole_stat_arr[-1]["stat"][effect_name] = {
                "N of spikes": sp.size,
                "Mean frequency": meanFr,
                "CV": cv,
                "Tau by maximums": tau_maxs,
                "Mode frequency": modeFr
            }
        """           

        
        fig_of_rate.set_size_inches(50, 5)
        
        fig_of_rate.savefig(neuron_dir + "rate_plot.png", dpi=500)
        plt.close("all")
        #break
    #break


#save_as_xlsx(whole_stat_arr, result_path + "whole_stat.xlsx")
#significance_stat = pyexcel.Sheet(effects_significance)
#significance_stat.save_as(result_path + "significance_of_effects.xlsx")