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

# Define data folder
data_folder = Path.home() / 'Documents' / 'data' / time.strftime("%Y%m%d")
# Try to create directory
try:
    os.makedirs(data_folder, exist_ok=True)
    print(f'Directory {data_folder} has been created successfully.')
except Exception as e:
    print(f'Error creating directory {data_folder}: {e}')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(data_folder, time.strftime("%H%M%S") + 'app.log'),
                    filemode='w')

config = configurations.MicroscopeConfiguration()


class MicroscopeGUI(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.module = module_main.MainModule(config, logging, data_folder)
        self.process = process_main.MainProcess(config, logging, data_folder)
        self.view = widget_main.MainWidget(config, logging, data_folder)
        self.controller = controller_main.MainController(self.view, self.module, self.process, config, logging,
                                                         data_folder)


def close():
    gui.module.close()
    app.exit()


app = QtWidgets.QApplication(sys.argv)
gui = MicroscopeGUI()
gui.view.show()

gui.view.Signal_quit.connect(close)

sys.exit(app.exec_())
