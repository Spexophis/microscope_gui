import os
import sys
import time
from pathlib import Path

from PyQt5 import QtWidgets

from controllers import controller_main
from modules import module_main
from processes import process_main
from utilities import configurations, error_log
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

        self.log_file = os.path.join(self.data_folder, time.strftime("%H%M%S") + 'app.log')
        self.info_log = error_log.ErrorLog(self.log_file)

        self.config = configurations.MicroscopeConfiguration(Path.home() / 'Documents' / 'data')

        try:
            self.module = module_main.MainModule(self.config, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")
        try:
            self.process = process_main.MainProcess(self.config, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")
        try:
            self.view = widget_main.MainWidget(self.config, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")
        try:
            self.controller = controller_main.MainController(self.view, self.module, self.process,
                                                             self.config, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")


def close():
    gui.module.close()
    app.exit()


app = QtWidgets.QApplication(sys.argv)
gui = MicroscopeGUI()
gui.view.show()

gui.view.Signal_quit.connect(close)

sys.exit(app.exec_())
