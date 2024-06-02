import numpy as np
import tifffile as tf


class ImageReconstruction:

    def __init__(self):
        self.na = 1.4
        self.wl = 0.5
        self.n, self.nx, self.ny = 27 * 27, 1024, 1024
        self.dx, self.dy = 0.0785, 0.0785
        self.dfx, self.dfy = 1 / (self.nx * self.dx), 1 / (self.ny * self.dy)
        self.rbx, self.rby = (self.na / self.wl) / self.dfx, (self.na / self.wl) / self.dfy
        self.psf = self.get_psf()

    def pupil_mask(self):
        x, y = self.meshgrid()
        msk = (x * x / (self.rbx * self.rbx)) + (y * y / (self.rby * self.rby)) <= 1
        msk = msk * 1
        phi = np.zeros((self.nx, self.ny))
        return msk * np.exp(1j * phi)

    def get_psf(self):
        bpp = self.pupil_mask()
        psf1 = np.abs((np.fft.fft2(np.fft.fftshift(bpp)))) ** 2
        return psf1 / psf1.sum()

    def meshgrid(self):
        x = np.arange(-self.nx / 2, self.nx / 2)
        y = np.arange(-self.ny / 2, self.ny / 2)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        return np.roll(xv, int(self.nx / 2)), np.roll(yv, int(self.ny / 2))
