import numpy as np
from PyQt5.QtCore import pyqtSlot


class ViewController:

    def __init__(self, view):
        self.view = view

    @pyqtSlot(np.ndarray)
    def plot_main(self, data):
        self.view.setImage('Main Camera', data)

    @pyqtSlot(np.ndarray)
    def plot_fft(self, data):
        self.view.setImage('FFT', data)

    @pyqtSlot(np.ndarray)
    def plot_sh(self, data):
        self.view.setImage('ShackHartmann', data)

    @pyqtSlot(np.ndarray)
    def plot_wf(self, data):
        self.view.setImage('Wavefront', data)

    @pyqtSlot(np.ndarray)
    def plot(self, data):
        self.view.plot(data)

    @pyqtSlot(np.ndarray)
    def plot_update(self, data):
        self.view.update_plot(data)
