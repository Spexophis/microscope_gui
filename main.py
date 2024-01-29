import os
import sys
import time
from pathlib import Path

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

        self.data_folder = Path.home() / 'Documents' / 'data' / time.strftime("%Y%m%d")
        try:
            os.makedirs(self.data_folder, exist_ok=True)
            print(f'Directory {self.data_folder} has been created successfully.')
        except Exception as e:
            print(f'Error creating directory {self.data_folder}: {e}')

        self.log_file = os.path.join(self.data_folder, time.strftime("%H%M%S") + 'app.log')
        self.info_log = error_log.ErrorLog(self.log_file)

        try:
            self.config = configurations.MicroscopeConfiguration(config_file)
        except Exception as e:
            self.info_log.error_log.error(f"Error: {e}")
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

    def init_config(self):
        self.setWindowTitle('Configuration File Selector')
        self.setGeometry(100, 100, 200, 100)
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        btn_open_dialog = cw.pushbutton_widget(r"Open Config File")
        btn_open_dialog.clicked.connect(self.open_config)
        layout.addWidget(btn_open_dialog)


def close():
    gui.module.close()
    app.exit()


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(''' 
                QDialog {
                    background-color: #333;
                    color: #FFF;
                }
                ''')
        self.init()
        self.selected_config = None

    def init(self):
        self.setWindowTitle('Select Configuration File')
        self.setGeometry(100, 100, 300, 100)
        layout = QtWidgets.QVBoxLayout(self)
        btn_select_file = cw.pushbutton_widget('Select File')
        btn_select_file.clicked.connect(self.open_file_dial)
        layout.addWidget(btn_select_file)

    def open_file_dial(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        title = "Select Configuration File"
        directory = str(Path.home() / "Documents" / "data" / "config_files")
        config_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, title, directory, "All Files (*)",
                                                                    options=options)
        if config_file_path:
            self.selected_config = config_file_path
            self.accept()


app = QtWidgets.QApplication(sys.argv)
dialog = ConfigDialog()
if dialog.exec_() == QtWidgets.QDialog.Accepted:
    cfd = dialog.selected_config
    if cfd:
        gui = MicroscopeGUI(cfd)
        gui.view.Signal_quit.connect(close)
        gui.view.show()
    else:
        QtWidgets.QMessageBox.information(None, "Information", "No configuration file selected. Application will exit.")
        sys.exit()
else:
    sys.exit()
sys.exit(app.exec_())
