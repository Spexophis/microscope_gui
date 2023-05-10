import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot


class ViewController:

    def __init__(self, view, module, process):
        self.v = view
        self.m = module
        self.p = process

    @pyqtSlot(np.ndarray)
    def plot_main(self, data):
        self.v.show_image('Main Camera', data)

    @pyqtSlot(np.ndarray)
    def plot_fft(self, data):
        self.v.show_image('FFT', data)

    @pyqtSlot(np.ndarray)
    def plot_sh(self, data):
        self.v.show_image('ShackHartmann', data)

    @pyqtSlot(np.ndarray)
    def plot_wf(self, data):
        self.v.show_image('Wavefront', data)

    @pyqtSlot(np.ndarray)
    def plot_dm(self, data):
        self.v.show_image('DM Calibration', data)

    def get_image_data(self, layer):
        return self.v.get_image(layer)

    @pyqtSlot(np.ndarray)
    def plot(self, data):
        self.v.plot(data)

    @pyqtSlot(np.ndarray)
    def plot_update(self, data):
        self.v.update_plot(data)

