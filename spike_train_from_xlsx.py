# -*- coding: utf-8 -*-
import os
import pyexcel
import pyexcel.ext.xlsx
import numpy as np
import spike_train_lib as stl
import matplotlib.pyplot as plt
###########################################################
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
    

        
    final_array = [[""], ["File name"]]
    for eff in effects:
        final_array[0].append(eff)
    for p in params:
        final_array[1].append(p) 

    for row_idx in range(n_rows):
        final_array.append([whole_stat_arr[row_idx]["File name"]])
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

###########################################################
# processing config
step = 10
hmsi_bin = 0.001
adiaphoriaFactor = 0.002 # 2 ms
###########################################################
# declare pathways to files
main_path = '/home/ivan/Data/speed_of_streem/Clampfit_Data_Pic/'
events_path = main_path + "events/"
if not ( os.path.isdir(events_path) ):
    raise SystemExit('Can not find source data directory!')

statistics_path = main_path + 'statistics/'
if not ( os.path.isdir(statistics_path) ):
    os.mkdir(statistics_path)
    
whole_stat_arr = []
    
for file_name in sorted(os.listdir(events_path)):
    
    if (file_name[0] == "."):
        continue
    print (file_name)
    saving_path = statistics_path + os.path.splitext(file_name)[0] + '/'
    
    if not ( os.path.isdir(saving_path) ):
        os.mkdir(saving_path)
    
    stat_dict = {"Effect name":[], "N of spikes":[], \
       "mean frequency":[], "cv":[], "tau_maxs":[] , "mode frequency":[]}
    statisticsByNeuron = file_name + \
    """
    Sheet name \t N of spikes \t mean frequency \t CV \t tau (maximums) \t mode frequency \n
    """
    whole_stat_arr.append({"File name":file_name, "stat":{} })
    
    file_path = events_path + file_name
    book = pyexcel.get_book(file_name=file_path)
    for sheet_name in sorted(book.sheet_names()):
        sheet = book.sheet_by_name(sheet_name)
                   
        spikes_list = []
        if (sheet.column[0][0] == 0):
            stat_dict["Effect name"].append(sheet_name)
            stat_dict["N of spikes"].append("-")
            stat_dict["mean frequency"].append("-")
            stat_dict["cv"].append("-")
            stat_dict["tau_maxs"].append("-")
            stat_dict["mode frequency"].append("-")
            statisticsByNeuron += ("%s \t - \t - \t - \t - \t - \n" % sheet_name)
            continue
        for el in sheet.column[4][1:]:
            spikes_list.append(el)
        
        sp = np.asarray(spikes_list) * 0.001
        # убираем повторные регистрации спайков
        sp = sp[ np.diff( np.append(0, sp) ) > adiaphoriaFactor ]        
        
        time_steps, rate = stl.get_rate_plot(sp, step)
        
        figOfRatePlot, axOfRatePlot = plt.subplots() 
        axOfRatePlot.step(time_steps[0:-1], rate)
        axOfRatePlot.set_title("Rate plot of " + sheet_name)
        axOfRatePlot.set_xlabel("time, sec")
        axOfRatePlot.set_ylabel("spike rate, sp/sec")
        axOfRatePlot.set_xlim(sp[0], sp[-1])
        axOfRatePlot.set_ylim(0, 1.2*rate.max())
        figOfRatePlot.savefig(saving_path + "Rate_plot_" + sheet_name, dpi=500)
        plt.show(block=False)
           
        hmsi_file = saving_path + "Hmsi_" + sheet_name
        hmsi_bins, hmsi, cv = stl.get_hmsi(sp, hmsi_bin, hmsi_file, sheet_name)
        
        auc_times, auc, tau_mins, tau_maxs  = stl.get_autororrelogram(sp, hmsi_bin)
        figOfAuc, axOfAuc = plt.subplots() 
                
        if ( np.sum(auc) > 0.001):
            axOfAuc.step(auc_times[0:-1], auc)
            axOfAuc.set_title("Autocorrelelogram of " + sheet_name)      
            axOfAuc.set_ylim(0, 1.2*np.max(auc) )
            figOfAuc.savefig(saving_path + "Autocorrelelogram_" + sheet_name, dpi=500)
            plt.show(block=False)
            
        frq, spectra, modeFr = stl.get_neuron_spectra(auc, auc_times[1]-auc_times[0] )
        
        if (np.sum(spectra) > 0.0001):
            figOfScr, axOfScr = plt.subplots() 
            axOfScr.step( frq, spectra )
            axOfScr.set_title("Neuron spectra of " + sheet_name) 
            figOfScr.savefig(saving_path + "Neuron spectra_" + sheet_name, dpi=500)
            meanFr = sp.size/( sp[-1] - sp[0] )
            plt.show(block=False)
    
        
        plt.close("all")
        statisticsByNeuron += ("""
                %s \t %i \t %f \t %f \t %f \t %f \n
                """ % (sheet_name, sp.size, meanFr, cv, tau_maxs, modeFr) )
    
        stat_dict["Effect name"].append(sheet_name)
        stat_dict["N of spikes"].append(sp.size)
        stat_dict["mean frequency"].append(meanFr)
        stat_dict["cv"].append(cv)
        stat_dict["tau_maxs"].append(tau_maxs)
        stat_dict["mode frequency"].append(modeFr)
        
        whole_stat_arr[-1]["stat"][sheet_name] = {
            "N of spikes": sp.size,
            "Mean frequency": meanFr,
            "CV": cv,
            "Tau by maximums": tau_maxs,
            "Mode frequency": modeFr
        }
        
    
    statistics_file = open(saving_path + 'stat.txt', "w")
    statistics_file.write (statisticsByNeuron)
    statistics_file.close()
    
    stat_xlsx = pyexcel.get_sheet(adict=stat_dict)
    stat_xlsx.save_as(saving_path + "stat.xlsx")
    
save_as_xlsx(whole_stat_arr, statistics_path + "whole_stat.xlsx")
