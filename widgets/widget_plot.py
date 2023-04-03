import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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
                
        Group_Plot = QtWidgets.QGroupBox('Plot')
        Group_Plot.setStyleSheet("font: bold Arial 10px")
        Group_Ctrl = QtWidgets.QGroupBox('Ctrl')
        Group_Ctrl.setStyleSheet("font: bold Arial 10px")
        
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
        
        self.QRadioButton_horizontal = QtWidgets.QRadioButton('Horizontal', self)
        self.QRadioButton_horizontal.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QRadioButton_vertical = QtWidgets.QRadioButton('Vertical', self)
        self.QRadioButton_vertical.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")        
        self.QPushButton_plot_static = QtWidgets.QPushButton('Plot', self)
        self.QPushButton_plot_static.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")        
        self.QPushButton_plot_update = QtWidgets.QPushButton('Update', self)
        self.QPushButton_plot_update.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
    
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
        