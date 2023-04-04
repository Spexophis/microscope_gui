from PIL import Image
import numpy as np
from ctypes import *
import copy

fnd = r'D:/winpython/python_code/microscope/slm_images'


class DeformableMirror:

    def __init__(self):
        super().__init__()

        Lcoslib = windll.LoadLibrary(r'Image_Control.dll')
