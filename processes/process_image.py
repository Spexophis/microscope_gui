import numpy as np


class ImageProcessing:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.wl = 0.5  # wavelength in microns
        self.na = 1.4  # numerical aperture
        self.dx = 13 / (2 * 5 * 63 / 3)  # pixel size in microns
        self.nx = 1024  # size of region
        self.fs = 1 / self.dx  # Spatial sampling frequency, inverse microns
        self.df = self.fs / self.nx  # Spacing between discrete frequency coordinates, inverse microns
        self.radius = (self.na / self.wl) / self.df

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def img_properties(self, img):
        return img.min(), img.max(), self.rms(img)

    @staticmethod
    def fourier_transform(data):
        return np.log(np.abs(np.fft.fftshift(np.fft.fft2(data))))

    @staticmethod
    def rms(data):
        nx, ny = data.shape
        n = nx * ny
        m = np.mean(data, dtype=np.float64)
        a = (data - m) ** 2
        r = np.sqrt(np.sum(a) / n)
        return r

    def snr(self, img, hpr):
        ny, nx = img.shape
        m = img.min()
        img[img <= m] = 0.
        img[img > m] = img[img > m] - m
        # w = self.circle(int(nx/4), nx)
        # img = img * w
        # lp = self.gaussianArr(shape=(nx,nx), sigma=lpr*self.radius, peakVal=1, orig=None, dtype=np.float32)
        # hp = 1-self.gaussianArr(shape=(nx,nx), sigma=hpr*self.radius, peakVal=1, orig=None, dtype=np.float32)
        r = self.radius
        lp = self.disc_array(shape=(nx, ny), radius=hpr * self.radius)
        hp = self.disc_array(shape=(nx, ny), radius=0.9 * self.radius) - self.disc_array(shape=(nx, ny), radius=hpr * self.radius)
        aft = np.fft.fftshift(np.fft.fft2(img))
        return (np.abs(hp * aft)).sum() / (np.abs(lp * aft)).sum()

    def hpf(self, img, hpr):
        nx, ny = img.shape
        m = img.min()
        img[img <= m] = 0.
        img[img > m] = img[img > m] - m
        # w = self.circle(int(nx/4), nx)
        # img = img*w
        hp = self.disc_array((nx, ny), self.radius) - self.disc_array((nx, ny), hpr * self.radius)
        # hp = 1-self.gaussianArr(shape=(nx,nx), sigma=hpr*self.radius, peakVal=1, orig=None, dtp=np.float32)
        aft = np.fft.fftshift(np.fft.fft2(img))
        aft = aft * hp
        return (np.abs(aft)).sum()

    @staticmethod
    def gaussian_filter(shape, sigma, peakVal, orig=None, dtp=np.float32):
        nx, ny = shape
        if orig is None:
            ux = nx / 2.
            uy = ny / 2.
        else:
            ux, uy = orig
        g = peakVal * np.fromfunction(lambda i, j: np.exp(-((i - ux) ** 2. + (j - uy) ** 2.) / (2. * sigma ** 2.)),
                                      (nx, ny))
        return g

    def peak(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        a, b, c = np.polyfit(x, y, 2)
        p = -1 * b / a / 2.0
        if a > 0:
            self.logg.error('no maximum')
            return 0.
        elif (p >= x.max()) or (p <= x.min()):
            self.logg.error('maximum exceeding range')
            return 0.
        else:
            return p

    @staticmethod
    def disc_array(shape=(128, 128), radius=64.0, origin=None, dtp=np.float64):
        nx = shape[0]
        ny = shape[1]
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        xv, yv = np.meshgrid(x, y)
        rho = np.sqrt(xv ** 2 + yv ** 2)
        disc = (rho < radius).astype(dtp)
        if origin is not None:
            s0 = origin[0] - int(nx / 2)
            s1 = origin[1] - int(ny / 2)
            disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
        return disc

    @staticmethod
    def get_profile(data, ax):
        data = data - data.min()
        data = data / data.max()
        if ax == 'X':
            return data.mean(0)
        if ax == 'Y':
            return data.mean(1)
