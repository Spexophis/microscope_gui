import json


class MicroscopeConfiguration:

    def __init__(self):
        file_path = r'C:/Users/Public/microscope_config'
        file_name = r'microscope_configurations.json'
        self.fd = file_path + r'/' + file_name

        self.config_data = {
            'control_matrix_phase': r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_phase_20230706.tif',
            'control_matrix_zonal': r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_zonal_20230706.tif',
            'control_matrix_modal': r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_modal_20230613.tif',
            'initial_flat': r'C:\Users\ruizhe.lin\Documents\data\dm_files\flatfile_20230627.xlsx',
        }

    def write_configurations(self):
        with open(self.fd, 'w') as f:
            json.dump(self.config_data, f)

    def read_configurations(self):
        with open(self.fd, 'r') as f:
            config = json.load(f)
        return config

    def change_configuration(self, item, value):
        with open(self.fd, 'w') as f:
            json.dump({item: value}, f)

    def read_configuration(self, item):
        with open(self.fd, 'r') as f:
            config = json.load(f)
        return config[item]
