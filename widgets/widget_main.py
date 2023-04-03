from PyQt5 import QtWidgets, QtCore

import widget_con
import widget_view
import widget_plot
import widget_ao

class MainWidget(QtWidgets.QMainWindow):
    
    Signal_quit = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        toolbar = QtWidgets.QToolBar()
        toolbar.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px; width: 36px; height: 18px")
        self.addToolBar(toolbar)

        button_exit = QtWidgets.QAction("Exit", self)
        button_exit.triggered.connect(self.Signal_quit.emit)
        toolbar.addAction(button_exit)

        self.con_view = widget_con.ConWidget()
        self.view_view = widget_view.ViewWidget()
        self.ao_view = widget_ao.AOWidget()
        self.plot_view = widget_plot.PlotWidget()
        
        self.dock_con = QtWidgets.QDockWidget('Dockable', self)
        self.dock_con.setWidget(self.con_view)
        self.dock_plot = QtWidgets.QDockWidget('Dockable', self)
        self.dock_plot.setWidget(self.plot_view)        
        self.dock_ao = QtWidgets.QDockWidget('Dockable', self)
        self.dock_ao.setWidget(self.ao_view)
        
        # set the view for the main window
        self.setCentralWidget(self.view_view)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_con)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_plot)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_ao)
      
        self.setWindowTitle("Microscope Control")
        self.setStyleSheet("background-color: #242424")

    def getControlWidget(self):
        return self.con_view

    def getViewWidget(self):
        return self.view_view
    
    def getAOWidget(self):
        return self.ao_view
    
    def getPlotWidget(self):
        return self.plot_view
    