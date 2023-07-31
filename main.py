import logging
import os
import sys
import time
from pathlib import Path

from PyQt5 import QtWidgets

from controllers import controller_main
from modules import module_main
from processes import process_main
from utilities import configurations
from widgets import widget_main


class MicroscopeGUI(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_folder = Path.home() / 'Documents' / 'data' / time.strftime("%Y%m%d")
        try:
            os.makedirs(self.data_folder, exist_ok=True)
            print(f'Directory {self.data_folder} has been created successfully.')
        except Exception as e:
            print(f'Error creating directory {self.data_folder}: {e}')

        self.logging.basicConfig(level=logging.INFO,
                                 format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 filename=os.path.join(self.data_folder, time.strftime("%H%M%S") + 'app.log'),
                                 filemode='w')

        self.config = configurations.MicroscopeConfiguration(Path.home() / 'Documents' / 'data')

        self.module = module_main.MainModule(self.config, self.logging, self.data_folder)
        self.process = process_main.MainProcess(self.config, self.logging, self.data_folder)
        self.view = widget_main.MainWidget(self.config, self.logging, self.data_folder)
        self.controller = controller_main.MainController(self.view, self.module, self.process,
                                                         self.config, self.logging, self.data_folder)


def close():
    gui.module.close()
    app.exit()


app = QtWidgets.QApplication(sys.argv)
gui = MicroscopeGUI()
gui.view.show()

gui.view.Signal_quit.connect(close)

sys.exit(app.exec_())
