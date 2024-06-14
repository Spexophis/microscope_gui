import sys

from PyQt5 import QtWidgets, QtCore

from miao.utilities import customized_widgets as cw
from miao.widgets import widget_ao, widget_con, widget_view


class MainWidget(QtWidgets.QMainWindow):
    Signal_quit = QtCore.pyqtSignal()

    def __init__(self, config, logg, path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.logg = logg.error_log
        self.data_folder = path

        self.con_view = widget_con.ConWidget(config, logg, path)
        self.view_view = widget_view.ViewWidget()
        self.ao_view = widget_ao.AOWidget(config, logg, path)

        self.dock_con = self.create_dock_widget(self.con_view)
        self.dock_ao = self.create_dock_widget(self.ao_view)

        self.dialog, self.dialog_text = cw.dialog(labtex=True)

        self.setCentralWidget(self.view_view)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_con)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_ao)

        self.dock_con.setFloating(True)
        self.dock_ao.setFloating(True)

        self.dock_con.hide()
        self.dock_ao.hide()

        self.setWindowTitle("Microscope Control")
        self.setStyleSheet("background-color: #121212; color: #FFFFFF")

        self.logg.info("Finish setting up widgets")

    def create_dock_widget(self, widget):
        dock_widget = QtWidgets.QDockWidget(self)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        dock_widget.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                                QtWidgets.QDockWidget.DockWidgetFloatable)
        dock_widget.setTitleBarWidget(CustomDockTitleBar(dock_widget))
        return dock_widget

    def closeEvent(self, event, **kwargs):
        self.Signal_quit.emit()
        self.con_view.save_spinbox_values()
        self.ao_view.save_spinbox_values()
        super().closeEvent(event)

    def get_dialog(self):
        self.dialog.exec_()
        self.dialog_text.setText(f"Task is running, please wait...")

    def get_file_dialog(self, sw="Save File"):
        file_dialog = cw.create_file_dialogue(name=sw, file_filter="All Files (*)", default_dir=self.data_folder)
        if file_dialog.exec_() == QtWidgets.QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()
            if selected_file:
                return selected_file[0]
            else:
                return None


class CustomDockTitleBar(QtWidgets.QWidget):
    def __init__(self, dock_widget):
        super().__init__()
        self.dock_widget = dock_widget
        self.init_ui()

    def init_ui(self):
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.label = QtWidgets.QLabel(self.dock_widget.windowTitle())
        self.layout.addWidget(self.label)

        self.minimize_button = QtWidgets.QToolButton()
        self.minimize_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TitleBarMinButton))
        self.minimize_button.clicked.connect(self.minimize_dock_widget)
        self.layout.addWidget(self.minimize_button)

    def minimize_dock_widget(self):
        self.dock_widget.setFloating(False)
        self.dock_widget.hide()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    config = None  # Replace with actual config
    logg = None  # Replace with actual logger
    path = ""  # Replace with actual path
    main_widget = MainWidget(config, logg, path)
    main_widget.show()
    sys.exit(app.exec_())
