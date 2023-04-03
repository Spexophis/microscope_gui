# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:46:54 2023

Inherited from GalvoScanDesigner.py in ImSwitch/imcontrol/model/signaldesigners

@author: Testa4
"""

import numpy as np
from scipy.interpolate import BPoly

class GalvoScan():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.bp_increase = BPoly.from_derivatives([0, 1], [[0., 0., 0.], [1., 0., 0.]])
        self.bp_decrease = BPoly.from_derivatives([0, 1], [[1., 0., 0.], [0., 0., 0.]])
