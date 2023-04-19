import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.style.use('dark_background')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from utilities import customized_widgets as cw
from utilities import napari_tools
from PyQt5 import QtWidgets, QtCore


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, dpi=512):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.set_facecolor("none")
        super(MplCanvas, self).__init__(fig)
        self.setStyleSheet("background-color: #242424")


class ViewWidget(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        dock_view, group_view = cw.create_dock('Camera View')
        dock_plot, group_plot = cw.create_dock('Plot')
        splitter.addWidget(dock_view)
        splitter.addWidget(dock_plot)
        layout.addWidget(splitter, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)

        layout_view = QtWidgets.QVBoxLayout()
        napari_tools.addNapariGrayclipColormap()
        self.napariViewer = napari_tools.EmbeddedNapari()
        layout_view.addWidget(self.napariViewer.get_widget(), alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        group_view.setLayout(layout_view)

        layout_plot = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas(self, dpi=100)
        toolbar = NavigationToolbar(self.canvas)
        layout_plot.addWidget(toolbar, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout_plot.addWidget(self.canvas, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        group_plot.setLayout(layout_plot)

        self.imgLayers = {}

        self.name_wf = 'Wavefront'
        self.imgLayers[self.name_wf] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_wf, blending='additive',
            colormap=None, protected=True)

        self.name_sh = 'ShackHartmann'
        self.imgLayers[self.name_sh] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_sh, blending='additive',
            colormap=None, protected=True)

        self.name_fft = 'FFT'
        self.imgLayers[self.name_fft] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_fft, blending='additive',
            colormap=None, protected=True)

        self.name_m = 'Main Camera'
        self.imgLayers[self.name_m] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_m, blending='additive',
            colormap=None, protected=True)

    def show_image(self, name, im):
        self.imgLayers[name].data = im

    def plot(self, data):
        self.canvas.axes.plot(data)
        self.canvas.draw()

    def update_plot(self, data):
        self.canvas.axes.cla()
        self.canvas.axes.plot(data)
        self.canvas.draw()


import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ViewWidget()
    window.show()
    sys.exit(app.exec_())

# Copyright (C) 2020-2021 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
