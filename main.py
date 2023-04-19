import sys

from PyQt5 import QtWidgets

from controllers import controller_main
from modules import module_main
from processes import process_main
from widgets import widget_main


class MicroscopeGUI(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = widget_main.MainWidget()
        self.process = process_main.MainProcess()
        self.module = module_main.MainModule()
        self.controller = controller_main.MainController(self.view, self.module, self.process)
        self.view.Signal_quit.connect(self.close)


def close():
    gui.module.close()
    gui.view.close()
    app.exit()


app = QtWidgets.QApplication(sys.argv)
gui = MicroscopeGUI()
gui.view.show()

gui.view.Signal_quit.connect(close)

sys.exit(app.exec_())
