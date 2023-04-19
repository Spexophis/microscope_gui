from PyQt5 import QtWidgets, QtCore

from utilities import customized_widgets as cw
from widgets import widget_ao
from widgets import widget_con
from widgets import widget_plot
from widgets import widget_view


class MainWidget(QtWidgets.QMainWindow):
    Signal_quit = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        toolbar = cw.toolbar_widget()
        self.addToolBar(toolbar)

        button_exit = QtWidgets.QAction("Exit", self)
        button_exit.triggered.connect(self.Signal_quit.emit)
        toolbar.addAction(button_exit)

        self.con_view = widget_con.ConWidget()
        self.view_view = widget_view.ViewWidget()
        self.ao_view = widget_ao.AOWidget()
        self.plot_view = widget_plot.PlotWidget()

        self.dock_con = cw.dock_widget()
        self.dock_con.setWidget(self.con_view)
        self.dock_plot = cw.dock_widget()
        self.dock_plot.setWidget(self.plot_view)
        self.dock_ao = cw.dock_widget()
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


import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = MainWidget()
    gui.show()
    sys.exit(app.exec_())
