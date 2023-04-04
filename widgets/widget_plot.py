import matplotlib

matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use('dark_background')
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from utilities import customized_widgets as cw


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, dpi=256):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.set_facecolor("none")
        super(MplCanvas, self).__init__(fig)
        self.setStyleSheet("background-color: #242424")


class PlotWidget(QtWidgets.QWidget):
    Signal_plot_static = QtCore.pyqtSignal()
    Signal_plot_update = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Layout = QtWidgets.QHBoxLayout(self)

        Group_Plot = cw.group_widget('Plot')
        Group_Ctrl = cw.group_widget('Ctrl')

        self.Layout.addWidget(Group_Plot)
        self.Layout.addWidget(Group_Ctrl)

        self.setLayout(self.Layout)

        Layout_Plot = QtWidgets.QVBoxLayout()

        self.canvas = MplCanvas(self, dpi=100)
        toolbar = NavigationToolbar(self.canvas, self)

        Layout_Plot.addWidget(toolbar, 0)
        Layout_Plot.addWidget(self.canvas, 1)

        Group_Plot.setLayout(Layout_Plot)

        Layout_Ctrl = QtWidgets.QVBoxLayout()

        self.QRadioButton_horizontal = cw.radiobutton_widget('Horizontal')
        self.QRadioButton_vertical = cw.radiobutton_widget('Vertical')
        self.QPushButton_plot_static = cw.pushbutton_widget('Plot')
        self.QPushButton_plot_update = cw.pushbutton_widget('Update')

        Layout_Ctrl.addWidget(self.QRadioButton_horizontal, 0)
        Layout_Ctrl.addWidget(self.QRadioButton_vertical, 1)
        Layout_Ctrl.addWidget(self.QPushButton_plot_static, 2)
        Layout_Ctrl.addWidget(self.QPushButton_plot_update, 3)

        Group_Ctrl.setLayout(Layout_Ctrl)

        self.QPushButton_plot_static.clicked.connect(self.plot_static)
        self.QPushButton_plot_update.clicked.connect(self.plot_update)

    def plot_static(self):
        self.Signal_plot_static.emit()

    def plot_update(self):
        self.Signal_plot_update.emit()

# import sys
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     main = PlotWidget()
#     main.show()
#     sys.exit(app.exec_())
