from PyQt5 import QtWidgets, QtCore

from utilities import customized_widgets as cw
from widgets import widget_ao
from widgets import widget_con
from widgets import widget_view


class MainWidget(QtWidgets.QMainWindow):
    Signal_quit = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.con_view = widget_con.ConWidget()
        self.view_view = widget_view.ViewWidget()
        self.ao_view = widget_ao.AOWidget()

        self.dock_con = cw.dock_widget()
        self.dock_con.setWidget(self.con_view)
        title_bar_widget_con = QtWidgets.QWidget()
        title_bar_widget_con.setFixedHeight(0)
        self.dock_con.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.dock_con.setTitleBarWidget(title_bar_widget_con)
        self.dock_ao = cw.dock_widget()
        self.dock_ao.setWidget(self.ao_view)
        title_bar_widget_ao = QtWidgets.QWidget()
        title_bar_widget_ao.setFixedHeight(0)
        self.dock_ao.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.dock_ao.setTitleBarWidget(title_bar_widget_ao)

        self.setCentralWidget(self.view_view)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_con)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_ao)

        self.setWindowTitle("Microscope Control")
        self.setStyleSheet("background-color: #242424")

    def closeEvent(self, event):
        self.Signal_quit.emit()
        print("Turning off the microscope")
        super().closeEvent(event)

    def get_control_widget(self):
        return self.con_view

    def get_view_widget(self):
        return self.view_view

    def get_ao_widget(self):
        return self.ao_view


import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = MainWidget()
    gui.show()
    sys.exit(app.exec_())
