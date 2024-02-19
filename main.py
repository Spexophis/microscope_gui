import os
import sys
import time

from PyQt5 import QtWidgets

from controllers import controller_main
from modules import module_main
from processes import process_main
from utilities import configurations, error_log
from utilities import customized_widgets as cw
from widgets import widget_main


class MicroscopeGUI(QtWidgets.QMainWindow):

    def __init__(self, config_file, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.config = configurations.MicroscopeConfiguration(config_file)
        except Exception as e:
            self.error_n_exit(f"Error loading configuration: {e}")
            return

        self.data_folder = self.config.configs["Data Path"] + r"\\" + time.strftime("%Y%m%d")
        try:
            os.makedirs(self.data_folder, exist_ok=True)
            print(f'Directory {self.data_folder} has been created successfully.')
        except Exception as e:
            print(f'Error creating directory {self.data_folder}: {e}')

        self.log_file = os.path.join(self.data_folder, time.strftime("%H%M%S") + 'app.log')
        self.info_log = error_log.ErrorLog(self.log_file)

        try:
            self.module = module_main.MainModule(self.config.configs, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")
        try:
            self.process = process_main.MainProcess(self.config.configs, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")
        try:
            self.view = widget_main.MainWidget(self.config.configs, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")
        try:
            self.controller = controller_main.MainController(self.view, self.module, self.process,
                                                             self.config.configs, self.info_log, self.data_folder)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")

    def error_n_exit(self, message):
        msg_box = cw.message_box("Error")
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.exec_()
        self.close()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    def close():
        gui.module.close()
        app.exit()


    cfd = r"C:\Users\ruizhe.lin\Documents\data\config_files\microscope_configurations_20240207.json"
    if cfd:
        try:
            gui = MicroscopeGUI(cfd)
            gui.view.Signal_quit.connect(close)
            gui.view.show()
        except Exception as er:
            print(f"Fatal error: {er}")
            sys.exit(1)

    sys.exit(app.exec_())
