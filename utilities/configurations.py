import json


class MicroscopeConfiguration:

    def __init__(self, fd=None):
        self.fd = fd
        if fd:
            self.configs = self.load_config()
        else:
            raise AttributeError(f"Configuration File Failed to Load")

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
