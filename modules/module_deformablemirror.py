import sys

sys.path.append(r'C:\Program Files\Alpao\sourcefiles')
from Lib64.asdk import DM
import numpy as np
import csv


class DeformableMirror:

    def __init__(self):

        try:
            serialName = 'BAX513'
            self.dm = DM(serialName)
            print("Connect the mirror")
            print("Retrieve number of actuators")
            self.nbAct = int(self.dm.Get('NBOfActuator'))
            print("Number of actuator for " + serialName + ": " + str(self.nbAct))
            self.z2c = self.zernike_modes()
        except:
            print('No DM found')

    def reset_dm(self):
        print("Reset")
        self.dm.Reset()
        print("Exit")

    def set_dm(self, values):
        if all(np.abs(v) < 1. for v in values):
            self.dm.Send(values)
            print('DM set')
        else:
            print("Some actuators exceed the DM push range!")
            pass

    def null_dm(self):
        self.dm.Send([0.] * self.nbAct)
        print('DM set to null')

    def zernike_modes(self):
        Z2C = []
        with open(r'C:/Program Files/Alpao/sourcefiles/BAX513-Z2C.csv', newline='') as csvfile:
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
