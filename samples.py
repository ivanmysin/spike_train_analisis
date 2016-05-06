# -*- coding: utf-8 -*-
"""

Ach septum processing

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


#######################################################
class SaveBounds:
    def __init__(self, line1, line2):
        self.bound1 = np.array([]) 
        self.bound2 = np.array([])
        self.line1 = line1
        self.line2 = line2
    
    def add(self, event):
        self.bound1 = np.append(self.bound1, line1.get_xdata()[0]) 
        self.bound2 = np.append(self.bound2, line2.get_xdata()[0]) 

    def save(self, event):
        for ind in range(self.bound1.size):
            print ( self.bound1[ind], self.bound2[ind] )

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
#########################################################
freqs = 8
t = np.linspace(0, 1, 1000)
s = np.cos( 2*np.pi*freqs*t )

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.2)
ax.set_title('Тестируем события в матплолиб!')
l, = ax.plot(t, s, lw=2)
line1,  = ax.plot([0.5, 0.5], [-1, 1], 'g')
line2,  = ax.plot([0.7, 0.7], [-1, 1], 'r')
linebuilder1 = LineBuilder(line1, "left_bound")
linebuilder2 = LineBuilder(line2, "right_bound")
plt.show()



callback = SaveBounds(line1, line2)
add = plt.axes([0.7, 0.05, 0.1, 0.075])
save = plt.axes([0.81, 0.05, 0.1, 0.075])
add_button = Button(add, 'Add')
add_button.on_clicked(callback.add)

save_button = Button(save, 'Save')
save_button.on_clicked(callback.save)


