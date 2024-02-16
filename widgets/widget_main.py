from PyQt5 import QtWidgets, QtCore

from utilities import customized_widgets as cw
from widgets import widget_ao, widget_con, widget_view


class MainWidget(QtWidgets.QMainWindow):
    Signal_quit = QtCore.pyqtSignal()

    def __init__(self, config, logg, path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.logg = logg
        self.data_folder = path

        self.con_view = widget_con.ConWidget(config, logg, path)
        self.view_view = widget_view.ViewWidget()
        self.ao_view = widget_ao.AOWidget(config, logg, path)

        self.dock_con = cw.dock_widget()
        self.dock_con.setWidget(self.con_view)
        title_bar_widget_con = QtWidgets.QWidget()
        title_bar_widget_con.setFixedHeight(0)
        self.dock_con.setTitleBarWidget(title_bar_widget_con)
        self.dock_ao = cw.dock_widget()
        self.dock_ao.setWidget(self.ao_view)
        title_bar_widget_ao = QtWidgets.QWidget()
        title_bar_widget_ao.setFixedHeight(0)
        self.dock_ao.setTitleBarWidget(title_bar_widget_ao)

        self.dialog, self.dialog_text = cw.dialog(labtex=True)

        self.setCentralWidget(self.view_view)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_con)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_ao)

        self.setWindowTitle("Microscope Control")
        self.setStyleSheet("background-color: #242424")

        self.logg.error_log.info("Finish setting up widgets")

    def closeEvent(self, event, **kwargs):
        self.Signal_quit.emit()
        super().closeEvent(event)

    def get_dialog(self):
        self.dialog.exec_()
        self.dialog_text.setText(f"Task is running, please wait...")


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = MainWidget(config=None, logg=None, path=None)
    gui.show()
    sys.exit(app.exec_())
