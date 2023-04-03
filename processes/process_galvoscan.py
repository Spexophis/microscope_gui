import numpy as np
from scipy.interpolate import BPoly

class GalvoScan():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.bp_increase = BPoly.from_derivatives([0, 1], [[0., 0., 0.], [1., 0., 0.]])
        self.bp_decrease = BPoly.from_derivatives([0, 1], [[1., 0., 0.], [0., 0., 0.]])
