import sys
from modules import module_main
from widgets import widget_main
from controllers import controller_main
from processes import process_main


class MicroscopeGUI:

    def __init__(self):
        super().__init__()

        self.viewer = widget_main.MainWidget()
        self.process = process_main.MainProcess()
        self.module = module_main.MainModule()
        self.controller = controller_main.MainController(self.my_view, self.module, self.process)
        self.viewer.Signal_quit.connect(self.close)

    def close(self):
        self.module.close()
        self.my_view.close()
        sys.exit()


if __name__ == '__main__':
    gui = MicroscopeGUI()
    gui.my_view.show()
