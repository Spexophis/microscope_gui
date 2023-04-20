import numpy as np


class ImageProcessing:

    def __init__(self):
        self.wl = 0.505  # wavelength in microns
        self.na = 1.4  # numerical aperture
        self.dx = 13 / (2.5 * 63)  # pixel size in microns
        self.nx = 1024  # size of region
        self.fs = 1 / self.dx  # Spatial sampling frequency, inverse microns
        self.df = self.fs / self.nx  # Spacing between discrete frequency coordinates, inverse microns
        self.radius = (self.na / self.wl) / self.df
        # self.dp = 1/(self.nx*self.dx) # pixel size in frequency space (pupil)
        # self.radius = (self.na/self.wl)/self.dp

    def wf_properties(self, wf):
        return wf.min(), wf.max(), self.rms(wf)

    def fourier_transform(self, data):
        return np.log(np.abs(np.fft.fftshift(np.fft.fft2(data))))

    def rms(self, data):
        nx, ny = data.shape
        n = nx * ny
        m = np.mean(data, dtype=np.float64)
        a = (data - m) ** 2
        r = np.sqrt(np.sum(a) / n)
        return r

    def snr(self, img, hpr):
        nx, ny = img.shape
        m = img.min()
        img[img <= m] = 0.
        img[img > m] = img[img > m] - m
        # w = self.circle(int(nx/4), nx)
        # img = img*w
        # lp = self.gaussianArr(shape=(nx,nx), sigma=lpr*self.radius, peakVal=1, orig=None, dtype=np.float32)
        # hp = 1-self.gaussianArr(shape=(nx,nx), sigma=hpr*self.radius, peakVal=1, orig=None, dtype=np.float32)
        lp = self.discArray((nx, ny), hpr * self.radius)
        hp = self.discArray((nx, ny), 0.9 * self.radius) - self.discArray((nx, ny), hpr * self.radius)
        aft = np.fft.fftshift(np.fft.fft2(img))
        return (np.abs(hp * aft)).sum() / (np.abs(lp * aft)).sum()

    def hpf(self, img, hpr):
        nx, ny = img.shape
        m = img.min()
        img[img <= m] = 0.
        img[img > m] = img[img > m] - m
        # w = self.circle(int(nx/4), nx)
        # img = img*w
        hp = self.discArray((nx, ny), 0.9 * self.radius) - self.discArray((nx, ny), hpr * self.radius)
        # hp = 1-self.gaussianArr(shape=(nx,nx), sigma=hpr*self.radius, peakVal=1, orig=None, dtype=np.float32)
        aft = np.fft.fftshift(np.fft.fft2(img))
        aft = aft * hp
        return (np.abs(aft)).sum()

    def peakv(self, img):
        nx, ny = img.shape
        # w = self.discArray((nx, ny), 64)
        # img = img*w
        maxv = img.max()
        return maxv

    def gaussianArr(self, shape, sigma, peakVal, orig=None, dtype=np.float32):
        nx, ny = shape
        if orig == None:
            ux = nx / 2.
            uy = ny / 2.
        else:
            ux, uy = orig
        g = np.fromfunction(lambda i, j: np.exp(-((i - ux) ** 2. + (j - uy) ** 2.) / (2. * sigma ** 2.)), (nx, ny),
                            dtype=dtype) * peakVal
        return g

    def peak(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        a, b, c = np.polyfit(x, y, 2)
        zmax = -1 * b / a / 2.0
        if (a > 0):
            print('no maximum')
            return 0.
        elif (zmax >= x.max()) or (zmax <= x.min()):
            print('maximum exceeding range')
            return 0.
        else:
            return zmax

    def discArray(self, shape=(128, 128), radius=64, origin=None, dtype=np.float64):
        nx = shape[0]
        ny = shape[1]
        ox = nx / 2
        oy = ny / 2
        x = np.linspace(-ox, ox - 1, nx)
        y = np.linspace(-oy, oy - 1, ny)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        disc = (rho < radius).astype(dtype)
        if not origin == None:
            s0 = origin[0] - int(nx / 2)
            s1 = origin[1] - int(ny / 2)
            disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
        return disc

    def get_profile(self, data, ax):
        data = data - data.min()
        data = data / data.max()
        if ax == 'X':
            return data.mean(0)
        if ax == 'Y':
            return data.mean(1)
