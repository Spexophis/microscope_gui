from PyQt5 import QtWidgets

import sys

from modules import module_main
from widgets import widget_main
from controllers import controller_main
from processes import process_main


class MicroscopeGUI(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.my_view = widget_main.MainWidget()

        self.process = process_main.MainProcess()

        self.module = module_main.Microscope()

        self.main_controller = controller_main.MainController(self.my_view, self.module, self.process)


app = QtWidgets.QApplication(sys.argv)
# app.setStyleSheet(
#     "QMainWindow {background-color: #383838; color: #F8F8F8} QPushButton {background-color: #636363; color: #F8F8F8} QDialog {background-color: #383838; color: #F8F8F8}")

gui = MicroscopeGUI()
gui.my_view.show()


def close():
    gui.module.close()
    gui.my_view.close()
    app.exit()


gui.my_view.Signal_quit.connect(close)

sys.exit(app.exec_())
