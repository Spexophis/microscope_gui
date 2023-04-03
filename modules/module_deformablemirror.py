# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:20:36 2022

@author: ruizhe.lin
"""

import os
import sys
sys.path.append(r'D:\aoresolft\Lib64')
from Lib64.asdk import DM
import numpy as np
import csv


class DeformableMirror():

    def __init__(self):
        super().__init__()

        try:
            self.serialName = 'BAX513'
            self.dm = DM(self.serialName)
            print("Connect the mirror")
            print("Retrieve number of actuators")
            self.nbAct = int(self.dm.Get('NBOfActuator'))
            print("Number of actuator for " + self.serialName + ": " + str(self.nbAct))
            # self.values = [0.] * self.nbAct
            # self.dm.Send( self.values )
        except:
            print('No DM found')

    def ResetDM(self):
        print("Reset")
        self.dm.Reset()
        print("Exit")

    def SetDM(self, values):
        if (all(np.abs(v) < 1. for v in values)):
            self.dm.Send(values)
            print('DM set')
        else:
            print("Some actuators exceed the DM push range!")
            pass

    def NullDM(self):
        self.values = [0.] * self.nbAct
        self.dm.Send(self.values)
        print('DM set to null')

    def SaveDM(self, pth, t, cmd):
        fns = '_' + t + 'dmcmd.csv'
        fns = os.path.join(pth, fns)
        with open(fns, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(cmd)

    # def SaveDM(self, pth, t, cmd):
    #     fns = '_' + t + 'dmcmd.txt'
    #     fns = os.path.join(pth,fns)
    #     with open(fns, 'w') as file:
    #         file.write(str(cmd))

    # def writeDMfile(self, pth, cmd, mod, zmv, re):
    #     fns = '_flatfile.txt'
    #     fns = os.path.join(pth,fns)
    #     with open(fns, 'w') as file:
    #         file.write(str(cmd))
    #     fn = '_modesvalue.txt'
    #     fn = os.path.join(pth, fn)
    #     np.savetxt(fn, (mod,zmv))
    #     fns1 = '_metric.csv'
    #     fns1 = os.path.join(pth, fns1)
    #     with open(fns1, "w", encoding='utf-8', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerows(re)

    def writeDMfile(self, pth, t, cmd, mod, zmv, re):
        filename = '_' + t + 'flatfile.csv'
        filename = os.path.join(pth, filename)
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(cmd)
        fn = '_' + t + '_modesvalue.csv'
        fn = os.path.join(pth, fn)
        with open(fn, "w") as f:
            writer = csv.writer(f)
            writer.writerow(mod)
            writer.writerow(zmv)
        fns = '_' + t + '_metric.csv'
        fns = os.path.join(pth, fns)
        with open(fns, "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(re)
