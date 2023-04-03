# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:43:30 2022

@author: Testa4
"""

from PIL import Image
import numpy as np
from ctypes import *
import copy

fnd = r'D:/winpython/python_code/microscope/slm_images'

class DeformableMirror():
    
    def __init__(self):
        super().__init__()
        
        Lcoslib = windll.LoadLibrary(r'Image_Control.dll')
        
        