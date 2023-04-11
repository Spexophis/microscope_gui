import sys

sys.path.append(r'C:\Program Files\Alpao\sourcefiles')
from Lib64.asdk import DM
import numpy as np


class DeformableMirror:

    def __init__(self):

        try:
            serialName = 'BAX513'
            self.dm = DM(serialName)
            print("Connect the mirror")
            print("Retrieve number of actuators")
            self.nbAct = int(self.dm.Get('NBOfActuator'))
            print("Number of actuator for " + serialName + ": " + str(self.nbAct))
            # self.values = [0.] * self.nbAct
            # self.dm.Send( self.values )
        except:
            print('No DM found')

    def ResetDM(self):
        print("Reset")
        self.dm.Reset()
        print("Exit")

    def SetDM(self, values):
        if all(np.abs(v) < 1. for v in values):
            self.dm.Send(values)
            print('DM set')
        else:
            print("Some actuators exceed the DM push range!")
            pass

    def NullDM(self):
        self.values = [0.] * self.nbAct
        self.dm.Send(self.values)
        print('DM set to null')
