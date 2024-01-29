import json
import os


class MicroscopeConfiguration:

    def __init__(self, pth):
        file_name = r"microscope_configurations.json"
        self.fd = os.path.join(pth, file_name)
        self.configs = self.load_config()

    def write_config(self, dataframe):
        with open(self.fd, 'w') as f:
            json.dump(dataframe, f, indent=4)

    def load_config(self):
        with open(self.fd, 'r') as f:
            cfg = json.load(f)
        return cfg

    def change_config(self, item, value):
        with open(self.fd, 'w') as f:
            json.dump({item: value}, f)
