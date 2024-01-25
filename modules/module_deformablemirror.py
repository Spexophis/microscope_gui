import csv
import struct
import sys, os

import numpy as np
import tifffile as tf
import pandas as pd
from tools import tool_zernike as tz
from tools import tool_improc as ipr

sys.path.append(r'C:\Program Files\Alpao\SDK\Samples\Python3')
if (8 * struct.calcsize("P")) == 32:
    from Lib.asdk import DM
else:
    from Lib64.asdk import DM


class DeformableMirror:

    def __init__(self, serial_name="BAX513", logg=None):
        self.logg = logg or self.setup_logging()
        self.dm, self.n_actuator = self._initialize_dm(serial_name)
        if self.dm is not None:
            self._configure_dm()

    def __del__(self):
        pass

    def _initialize_dm(self, sn):
        try:
            dm = DM(sn)
            n_act = int(dm.Get('NBOfActuator'))
            self.logg.info("Number of actuator for " + sn + ": " + str(n_act))
            return dm, n_act
        except Exception as e:
            self.logg.error(f"Error Initializing DM: {e}")
            return None, None

    def _configure_dm(self):
        path = r"C:\Users\ruizhe.lin\Documents\data\dm_files\bax513"
        influence_function_images = tf.imread(os.path.join(path, "influence_function_images_20240124.tif"))
        nct, self.nly, self.nlx = influence_function_images.shape
        self.nls = self.nly * self.nlx
        self.control_matrix_phase = tf.imread(os.path.join(path, "control_matrix_phase_20240124.tif"))
        self.control_matrix_zonal = tf.imread(os.path.join(path, "control_matrix_zonal_20240124.tif"))
        self.control_matrix_modal = tf.imread(os.path.join(path, "control_matrix_modal_20230706.tif"))
        initial_flat = os.path.join(path, "flatfile_20230728.xlsx")
        if self.control_matrix_phase.shape[0] != self.n_actuator:
            self.logg.error(f"Wrong size of DM control matrix")
        self.dm_cmd = [[0.] * self.n_actuator]
        self.dm_cmd.append(self.read_cmd(initial_flat))
        self.current_cmd = 1
        try:
            self.set_dm(self.dm_cmd[self.current_cmd])
        except Exception as e:
            self.logg.error(f"Error set dm {e}")
        self.correction = []
        self.temp_cmd = []
        self.amp = 0.1
        self.n_zernike = 60
        self.az = None
        self.zernike = tz.get_zernike_polynomials(nz=self.n_zernike, size=[self.nly, self.nlx])
        self.zslopes = tz.get_zernike_slopes(nz=self.n_zernike, size=[self.nly, self.nlx])
        # self.z2c = self.zernike_modes()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def close(self):
        self.reset_dm()
        self.logg.info("Exit")

    def reset_dm(self):
        self.dm.Reset()
        self.logg.info("Reset")

    def set_dm(self, values):
        if all(np.abs(v) < 1. for v in values):
            self.dm.Send(values)
            self.logg.info('DM set')
        else:
            raise ValueError("Some actuators exceed the DM push range!")

    def null_dm(self):
        self.dm.Send([0.] * self.n_actuator)
        self.logg.info('DM set to null')

    def get_correction(self, measurements, method="phase"):
        if method == 'phase':
            self.correction.append(list(self.amp * np.dot(self.control_matrix_phase, -measurements.reshape(self.nls))))
        else:
            gradx, grady = measurements
            measurement = np.concatenate((gradx.reshape(self.nls), grady.reshape(self.nls)))
            if method == 'zonal':
                self.correction.append(list(np.dot(self.control_matrix_zonal, -measurement)))
            elif method == 'modal':
                a = ipr.get_eigen_coefficients(-measurement, self.zslopes)
                self.correction.append(list(np.dot(self.control_matrix_modal, a)))
            else:
                self.logg.error(f"Invalid AO correction method")
                return
        _c = self.cmd_add(self.dm_cmd[self.current_cmd], self.correction[-1])
        self.dm_cmd.append(_c)

    @staticmethod
    def zernike_modes():
        """
        z2c index:
        0, 1 - tip / tilt
        2 - defocus
        3, 4 - astigmatism
        5, 6 - coma
        7, 8 - trefoil
        9 - spherical
        """
        Z2C = []
        with open(r'C:\Program Files\Alpao\SDK\Config\BAX513-Z2C.csv', newline='') as csvfile:
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
            filename = t + "_flat_file.xlsx"
            df = pd.DataFrame(self.dm_cmd[-1], index=np.arange(97), columns=['Push'])
            df.to_excel(os.path.join(path, filename), index_label='Actuator')
        else:
            filename = t + "_cmd_file.xlsx"
            data = {f'cmd{i}': cmd for i, cmd in enumerate(self.dm_cmd)}
            with pd.ExcelWriter(os.path.join(path, filename), engine='xlsxwriter') as writer:
                for sheet_name, list_data in data.items():
                    df = pd.DataFrame(list_data, index=np.arange(97), columns=['Push'])
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
