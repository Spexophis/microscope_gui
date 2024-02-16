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
    Signal_image_metrics = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_ui()
        self._set_napari_layers()
        self._set_signal_connections()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self._create_docks()
        self._create_widgets()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        for name, (dock, group) in self.docks.items():
            splitter.addWidget(dock)
            group.setLayout(self.widgets[name])
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_docks(self):
        self.docks = {
            "view": cw.create_dock("Image"),
            "metric": cw.create_dock("Image Metrics"),
            "plot": cw.create_dock("Profile")
        }

    def _create_widgets(self):
        self.widgets = {
            "view": self._create_view_widgets(),
            "metric": self._create_metric_widgets(),
            "plot": self._create_plot_widgets()
        }

    def _create_view_widgets(self):
        layout_view = QtWidgets.QVBoxLayout()
        napari_tools.addNapariGrayclipColormap()
        self.napariViewer = napari_tools.EmbeddedNapari()
        layout_view.addWidget(self.napariViewer.get_widget())
        return layout_view

    def _create_metric_widgets(self):
        layout_metric = QtWidgets.QGridLayout()
        self.QLabel_img_laplacian_cv2 = cw.label_widget(str('Laplacian (cv2)'))
        self.QLCDNumber_img_laplacian_cv2 = cw.lcdnumber_widget()
        self.QLabel_img_laplacian_scikit = cw.label_widget(str('Laplacian (scikit)'))
        self.QLCDNumber_img_laplacian_scikit = cw.lcdnumber_widget()
        self.QLabel_img_sobel_scikit = cw.label_widget(str('Sobel (scikit)'))
        self.QLCDNumber_img_sobel_scikit = cw.lcdnumber_widget()
        self.QPushButton_image_metrics = cw.pushbutton_widget('Calculate')
        layout_metric.addWidget(self.QLabel_img_laplacian_cv2, 0, 0, 1, 1)
        layout_metric.addWidget(self.QLCDNumber_img_laplacian_cv2, 1, 0, 1, 1)
        layout_metric.addWidget(self.QLabel_img_laplacian_scikit, 0, 1, 1, 1)
        layout_metric.addWidget(self.QLCDNumber_img_laplacian_scikit, 1, 1, 1, 1)
        layout_metric.addWidget(self.QLabel_img_sobel_scikit, 0, 2, 1, 1)
        layout_metric.addWidget(self.QLCDNumber_img_sobel_scikit, 1, 2, 1, 1)
        layout_metric.addWidget(self.QPushButton_image_metrics, 1, 3, 1, 1)
        return layout_metric

    def _create_plot_widgets(self):
        layout_plot = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas(self, dpi=64)
        toolbar = NavigationToolbar(self.canvas)
        layout_plot.addWidget(toolbar)
        layout_plot.addWidget(self.canvas)
        return layout_plot

    def _set_napari_layers(self):
        self.napari_layers = {}
        self.img_layers = {0: "Andor EMCCD", 1: "Hamamatsu sCMOS", 2: "DMK 33UX250", 3: "FFT", 4: "ShackHartmann(Base)",
                           5: "Wavefront"}
        for name in reversed(list(self.img_layers.values())):
            self.napari_layers[name] = self.add_napari_layer(name)

    def _set_signal_connections(self):
        self.QPushButton_image_metrics.clicked.connect(self.calculate_image_metric)

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

    @QtCore.pyqtSlot()
    def calculate_image_metric(self):
        self.Signal_image_metrics.emit()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ViewWidget()
    window.show()
    sys.exit(app.exec_())
