# -*- coding: utf-8 -*-
"""
burst analisis
"""
import numpy as np
import matplotlib.pyplot as plt
spike_train = 0.5*np.random.randn(5) + 10

for i in range(2, 10):
    new_burst = 0.5*np.random.randn(5) + 10*i
    spike_train = np.append(spike_train, new_burst)

spike_train = np.sort(spike_train)

#bursts = kleinberg(st)

isi = np.diff(spike_train)

sp = np.empty((2, spike_train.size), dtype=float)
sp[0, :] = np.append(np.mean(isi), isi)
sp[1, :] = np.append(isi, np.mean(isi) )


plt.scatter(sp[0, :], sp[1, :], s=30)

fd = 0.1
"""
level = (bursts[:, 0] >= 1)
firt_spikes = bursts[level, 1]
last_spikes = bursts[level, 2]
"""
spike = np.zeros( spike_train[-1]/fd + 1 )
spike[ np.floor(spike_train/fd).astype(int) ] = 1
time = np.linspace(0, spike_train[-1], spike.size)
      
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time, spike, linewidth=3)
#ax.scatter(firt_spikes, np.ones(firt_spikes.size), s=70, color="red")
#ax.scatter(last_spikes, np.ones(firt_spikes.size), s=70, color="green")
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 1.5)
