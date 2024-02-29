import csv
import os
import struct
import sys

import numpy as np
import pandas as pd
import tifffile as tf

from miao.tools import tool_improc as ipr
from miao.tools import tool_zernike as tz

sys.path.append(r'C:\Program Files\Alpao\SDK\Samples\Python3')
if (8 * struct.calcsize("P")) == 32:
    from Lib.asdk import DM
else:
    from Lib64.asdk import DM


class DeformableMirror:

    def __init__(self, name="ALPAO DM97", logg=None, config=None):
        self.logg = logg or self.setup_logging()
        self.config = config or self.load_configs()
        self.dm_name = name
        self.dm_serial = self.config["Adaptive Optics"]["Deformable Mirrors"][self.dm_name]["Serial"]
        self.dm, self.n_actuator = self._initialize_dm(self.dm_serial)
        if self.dm is not None:
            self._configure_dm()
        else:
            raise RuntimeError(f"Error Initializing DM {self.dm_name}")
        try:
            self.set_dm(self.dm_cmd[self.current_cmd])
        except Exception as e:
            self.logg.error(f"Error set dm {e}")

    def __del__(self):
        pass

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @staticmethod
    def load_configs():
        import json
        config_file = input("Enter configuration file directory: ")
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg

    def _initialize_dm(self, sn):
        try:
            dm = DM(sn)
            n_act = int(dm.Get('NBOfActuator'))
            self.logg.info("Number of actuator for " + sn + ": " + str(n_act))
            return dm, n_act
        except Exception as e:
            self.logg.error(f"Error Initializing DM {self.dm_name}: {e}")
            return None, None

    def _configure_dm(self):
        try:
            influence_function_images = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirrors"][self.dm_name]["Influence Function Images"])
            nct, self.nly, self.nlx = influence_function_images.shape
            self.nls = self.nly * self.nlx
            self.control_matrix_phase = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirrors"][self.dm_name]["Phase Control Matrix"])
            self.control_matrix_zonal = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirrors"][self.dm_name]["Zonal Control Matrix"])
            self.initial_flat = self.config["Adaptive Optics"]["Deformable Mirrors"][self.dm_name]["Initial Flat"]
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} files: {e}")
        try:
            self.control_matrix_modal = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirrors"][self.dm_name]["Modal Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} modal control file: {e}")
        if hasattr(self, "initial_flat"):
            self.dm_cmd = [[0.] * self.n_actuator]
            self.dm_cmd.append(self.read_cmd(self.initial_flat))
            self.current_cmd = 1
            self.correction = []
            self.temp_cmd = []
            self.amp = 0.1
            self.n_zernike = 60
            self.az = None
            self.zernike = tz.get_zernike_polynomials(nz=self.n_zernike, size=[self.nly, self.nlx])
            # self.zslopes = tz.get_zernike_slopes(nz=self.n_zernike, size=[self.nly, self.nlx])
            # self.z2c = self.zernike_modes()
        else:
            self.dm_cmd = [[0.] * self.n_actuator]
            self.current_cmd = 0
            self.correction = []
            self.temp_cmd = []
            self.amp = 0.1
            self.logg.error(f"Missing initial flat, started with Null")

    def close(self):
        self.reset_dm()
        self.logg.info(f"DM {self.dm_name} Close")

    def reset_dm(self):
        self.dm.Reset()
        self.logg.info(f"DM {self.dm_name} Reset")

    def set_dm(self, values):
        if all(np.abs(v) < 1. for v in values):
            self.dm.Send(values)
            self.logg.info(f"DM {self.dm_name} set")
        else:
            raise ValueError("Some actuators exceed the DM push range!")

    def null_dm(self):
        self.dm.Send([0.] * self.n_actuator)
        self.logg.info(f"DM {self.dm_name} set to null")

    def get_correction(self, measurements, method="phase"):
        if method == 'phase':
            self.correction.append(list(self.amp * np.dot(self.control_matrix_phase, -measurements.reshape(self.nls))))
        else:
            gradx, grady = measurements
            measurement = np.concatenate((gradx.reshape(self.nls), grady.reshape(self.nls)))
            if method == 'zonal':
                self.correction.append(list(np.dot(self.control_matrix_zonal, -measurement)))
            # elif method == 'modal':
            #     a = ipr.get_eigen_coefficients(-measurement, self.zslopes)
            #     self.correction.append(list(np.dot(self.control_matrix_modal, a)))
            else:
                self.logg.error(f"Invalid AO correction method")
                return
        _c = self.cmd_add(self.dm_cmd[self.current_cmd], self.correction[-1])
        self.dm_cmd.append(_c)

    def zernike_modes(self):
        """
        z2c index:
        0, 1 - tip / tilt
        2 - defocus
        3, 4 - astigmatism
        5, 6 - coma
        7, 8 - trefoil
        9 - spherical
        """
        pth = r"C:\Program Files\Alpao\SDK\Config"
        fn = f"{self.dm_serial}-Z2C.csv"
        Z2C = []
        with open(os.path.join(pth, fn), newline='') as csvfile:
            csvrows = csv.reader(csvfile, delimiter=' ')
            for row in csvrows:
                x = row[0].split(",")
                Z2C.append(x)
        for i in range(len(Z2C)):
            for j in range(len(Z2C[i])):
                Z2C[i][j] = float(Z2C[i][j])
        return Z2C

    def get_zernike_cmd(self, j, a):
        zerphs = a * self.zernike[j]
        return list(np.dot(self.control_matrix_phase, zerphs.reshape(self.nls)))

    @staticmethod
    def cmd_add(cmd_0, cmd_1):
        return list(np.asarray(cmd_0) + np.asarray(cmd_1))

    @staticmethod
    def read_cmd(fnd):
        df = pd.read_excel(fnd)
        return df['Push'].tolist()

    def write_cmd(self, path, t, flatfile=False):
        if flatfile:
            filename = t + f"{self.dm_serial}_flat_file.xlsx"
            df = pd.DataFrame(self.dm_cmd[-1], index=np.arange(self.n_actuator), columns=['Push'])
            fd = os.path.join(path, filename)
            df.to_excel(str(fd), index_label='Actuator')
        else:
            filename = t + f"{self.dm_serial}_cmd_file.xlsx"
            fd = os.path.join(path, filename)
            data = {f'cmd{i}': cmd for i, cmd in enumerate(self.dm_cmd)}
            with pd.ExcelWriter(str(fd), engine='xlsxwriter') as writer:
                for sheet_name, list_data in data.items():
                    df = pd.DataFrame(list_data, index=np.arange(self.n_actuator), columns=['Push'])
                    df.to_excel(writer, sheet_name=sheet_name, index_label='Actuator')

    def save_sensorless_results(self, fd, a, v, p):
        df1 = pd.DataFrame(v, index=a, columns=['Values'])
        df2 = pd.DataFrame(p, index=np.arange(self.n_zernike), columns=['Amplitudes'])
        with pd.ExcelWriter(fd, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Metric Values')
            df2.to_excel(writer, sheet_name='Peaks')

    def wavefront_zernike_reconstruct(self, wf, size=None, exclusive=True):
        return self.wavefront_recomposition(self.wavefront_decomposition(wf), size=size, exclusive=exclusive)

    def wavefront_decomposition(self, wf):
        za = np.zeros(self.n_zernike)
        for i in range(self.n_zernike):
            wz = self.zernike[i]
            za[i] = (wf * wz.conj()).sum() / (wz * wz.conj()).sum()
        return za

    def wavefront_recomposition(self, za, size=None, exclusive=True):
        if size is None:
            ny = self.nly
            nx = self.nlx
        else:
            ny, nx = size
        wf = np.zeros((ny, nx))
        if exclusive:
            za[:4] = 0
        for i in range(self.n_zernike):
            wf += za[i] * self.zernike[i]
        return wf
