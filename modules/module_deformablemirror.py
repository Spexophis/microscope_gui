import csv
import struct
import sys

import numpy as np

sys.path.append(r'C:\Program Files\Alpao\SDK\Samples\Python3')
if (8 * struct.calcsize("P")) == 32:
    from Lib.asdk import DM
else:
    from Lib64.asdk import DM


class DeformableMirror:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        try:
            serial_name = 'BAX513'
            self.dm = DM(serial_name)
            self.logg.info("Retrieve number of actuators")
            self.nbAct = int(self.dm.Get('NBOfActuator'))
            self.logg.info("Number of actuator for " + serial_name + ": " + str(self.nbAct))
            self.z2c = self.zernike_modes()
        except Exception as e:
            self.logg.error(f"Error Initializing DM: {e}")

    def __del__(self):
        pass

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
            self.logg.error("Some actuators exceed the DM push range!")
            pass

    def null_dm(self):
        self.dm.Send([0.] * self.nbAct)
        self.logg.info('DM set to null')

    @staticmethod
    def zernike_modes():
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


"""
z2c index:
0, 1 - tip / tilt
2 - defocus
3, 4 - astigmatism
5, 6 - coma
7, 8 - trefoil
9 - spherical
"""
