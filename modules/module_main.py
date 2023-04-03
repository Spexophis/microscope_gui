# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:27:48 2022

@author: ruizhe.lin
"""

import module_andorixon
import module_hamamatsu
# import module_tiscamera
# import module_thorcamera
import module_deformablemirror
import module_laser
import module_daq
import module_mcldeck


class Microscope():

    def __init__(self):
        self.cam = module_andorixon.EMCCDCamera()
        self.hacam = module_hamamatsu.HamamatsuCameraMR()
        self.dm = module_deformablemirror.DeformableMirror()
        self.laser = module_laser.CoboltLaser()
        self.daq = module_daq.DAQ()
        # self.tiscam = module_tiscamera.TISCamera()
        self.md = module_mcldeck.MCLMicroDrive()
        # self.thocam = module_thorcamera.UC480Cam()

    def close(self):
        self.cam.shutdown()
        self.hacam.shutdown()
        # self.tiscam.close()
        # self.thocam.close()
        self.laser.all_off()
        self.dm.ResetDM()
        self.daq.Reset_daq()
        self.md.close()
