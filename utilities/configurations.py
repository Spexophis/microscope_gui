import json

configs = {
    "microscope": {
        "objective_na": 1.4,
        "objective_magnification": 63,
        "overall_magnification": 168,
    },
    "camera_andor": {
        "pixel_size / um": 13,
        "em_gain": 50,
    },
    "camera_hamamatsu": {
        "pixel_size / um": 6.5,
    },
    "sh_wavefront_sensor": {
        "pixel_size / um": 6.5,
    },
    "dm_files": {
        "control_matrix_phase": r"C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_phase_20230706.tif",
        "control_matrix_zonal": r"C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_zonal_20230706.tif",
        "control_matrix_modal": r"C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_modal_20230613.tif",
        "initial_flat": r"C:\Users\ruizhe.lin\Documents\data\dm_files\flatfile_20230627.xlsx",
    },
}


def write_config(filename, dataframe):
    with open(filename, 'w') as f:
        json.dump(dataframe, f, indent=4)


def load_config(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg
