# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:58:45 2022

@author: Testa4
"""

class PlotController():

    def __init__(self, view):
        self.view = view
        self.canvas = view.canvas 
        
    def get_plot_axis(self):
        if self.view.QRadioButton_horizontal.isChecked():
            h = True
            v = False
            print('Horizontal')
        if self.view.QRadioButton_vertical.isChecked():
            h = False
            v = True
            print('Vertical')
        return h, v
        
    def plot_profile(self, img, h=True, v=False):
        data = img - img.min()
        data = data / data.max()
        if h:
            mean = data.mean(0)
        if v:
            mean = data.mean(1)
        self.canvas.axes.plot(mean, linewidth=1)
        self.canvas.draw()

    def updata_plot(self, img, h=True, v=False):
        self.canvas.axes.cla()
        data = img - img.min()
        data = data / data.max()
        if h:
            mean = data.mean(0)
        if v:
            mean = data.mean(1)
        self.canvas.axes.plot(mean, linewidth=1)
        self.canvas.draw()
        