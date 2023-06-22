import numpy as np
import tifffile as tf
from numpy.fft import fft2, fftshift
from skimage.restoration import unwrap_phase


class Interferometer:

    def __init__(self):
        pass

    def reconstruction_surface(self, img, msk):
        imf = np.fft.fftshift(np.fft.fft2(img * msk))
        cf = self.extract_order(imf)
        ph = np.fft.ifft2(cf)
        phase = np.arctan2(ph.imag, ph.real)
        phase_unwrapped = unwrap_phase(phase)
        return phase, phase_unwrapped

    def extract_order(self, df):
        coordinates = self.find_order(df)
        y, x = coordinates[0]
        ny, nx = df.shape

        return fftshift(df[y - hsy: y + hsy, x - hsx:x + hsx])



