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
        layout.addWidget(splitter)
        self.setLayout(layout)

        layout_view = QtWidgets.QVBoxLayout()
        napari_tools.addNapariGrayclipColormap()
        self.napariViewer = napari_tools.EmbeddedNapari()
        layout_view.addWidget(self.napariViewer.get_widget())
        group_view.setLayout(layout_view)

        layout_plot = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas(self, dpi=64)
        toolbar = NavigationToolbar(self.canvas)
        layout_plot.addWidget(toolbar)
        layout_plot.addWidget(self.canvas)
        group_plot.setLayout(layout_plot)

        self.napari_layers = {}
        self.img_layers = {0: "Andor EMCCD", 1: "Hamamatsu sCMOS", 2: "DMK 33UX250", 3: "FFT", 4: "ShackHartmann(Base)",
                           5: "Wavefront"}
        for name in reversed(list(self.img_layers.values())):
            self.napari_layers[name] = self.add_napari_layer(name)

    def add_napari_layer(self, name):
        return self.napariViewer.add_image(np.zeros((1024, 1024)), rgb=False, name=name, blending='additive',
                                           colormap=None, protected=True)

    def show_image(self, name, im):
        if isinstance(im, np.ndarray):
            self.napari_layers[name].data = im

    def get_image(self, name):
        return self.napari_layers[name].data

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
