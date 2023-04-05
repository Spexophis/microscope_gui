import asyncio
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from findiff import FinDiff
from scipy.ndimage import center_of_mass as com
from scipy.signal import fftconvolve as corr
from scipy.special import factorial
from skimage.filters import threshold_otsu

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
pi = np.pi


# influence_fuction
# flat_start


class WavefrontReconstruction:

    def __init__(self):
        self.radius = 8  # 1/2 the total number of lenslets in linear direction
        self.diameter = 16  # total number of lenslets in linear direction
        self.x_center_base = 256
        self.y_center_base = 256
        self.x_center_offset = 256
        self.y_center_offset = 256
        self.px_spacing = 16  # spacing between each lenslet
        self.hsp = 8  # size of subimage is 2*hsp
        self.calfactor = (.0046 / 5.2) * (150)  # pixel size * focalLength * pitch
        # set up seccorr center
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)

        self.base = np.zeros((512, 512))
        self.offset = np.zeros((512, 512))
        self.wf = np.zeros((self.diameter, self.diameter))

        self.zernike = self._zernike_polynomials(nz=58, size=[self.diameter, self.diameter])
        self._correction = []
        self._dm_cmd = []

    def _update_parameters(self, parameters):
        self.x_center_base = parameters[0]
        self.y_center_base = parameters[1]
        self.x_center_offset = parameters[2]
        self.y_center_offset = parameters[3]
        self.diameter = parameters[4]
        self.px_spacing = parameters[6]
        self.hsp = parameters[5]
        self.wf = np.zeros((self.diameter, self.diameter))
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)

    def _find_center(self, base, offset):
        secbs = base[self.y_center_base - self.hsp: self.y_center_base + self.hsp,
                self.x_center_base - self.hsp: self.x_center_base + self.hsp]
        secof = offset[self.y_center_offset - self.hsp: self.y_center_offset + self.hsp,
                self.x_center_offset - self.hsp: self.x_center_offset + self.hsp]
        ind_bas = com(secbs)
        ind_off = com(secof)
        self.x_center_base = self.x_center_base - self.hsp + round(ind_bas[1])
        self.y_center_base = self.y_center_base - self.hsp + round(ind_bas[0])
        self.x_center_offset = self.x_center_offset - self.hsp + round(ind_off[1])
        self.y_center_offset = self.y_center_offset - self.hsp + round(ind_off[0])

    def _generate_influence_function(self, data, method='zonal'):
        n, x, y = data.shape
        if n % 2 != 0:
            raise "The image number has to be even"
        self._n_actuators = int(n / 2)
        self._n_lenslets = self.diameter * self.diameter
        self._influence_function = np.zeros((2 * self._n_lenslets, self._n_actuators))
        for i in range(int(n / 2)):
            self._get_gradient_xy(data[i], data[i + 1])
            self._influence_function[:self._n_lenslets, i] = self.gradx.flatten()
            self._influence_function[self._n_lenslets:, i] = self.grady.flatten()

    def _get_control_matrix(self, method='zonal'):
        u, s, vh = np.linalg.svd(self._influence_function, full_matrices=True)
        self._control_matrix = np.dot(vh.T, np.dot(np.diag(np.diag(1 / s)), u.T))

    def _close_loop_correction(self, measurement, method='zonal'):
        self._get_gradient_xy(self.base, measurement)
        self._measurement = np.concatenate((self.gradx.flatten(), self.grady.flatten()))
        _c = np.dot(self._control_matrix, self._measurement)
        self._correction.append(_c)

    def _correct_cmd(self):
        _c = self._dm_cmd[-1] + self._correction
        self._dm_cmd.append(_c)

    def _get_gradient_xy(self, baseimg, offsetimg):
        """ Determines Gradients by Correlating each section with its base reference section"""
        self.nx = self.diameter
        self.ny = self.diameter
        self.im = np.zeros((2, 2 * self.hsp * self.diameter, 2 * self.hsp * self.diameter))
        self.gradxy = np.zeros((2, self.diameter, self.diameter))
        self.gradx = np.zeros((self.ny, self.nx))
        self.grady = np.zeros((self.ny, self.nx))
        # self.findcenter(baseimg, offsetimg)
        # baseimg = baseimg / baseimg.max()
        # offsetimg = offsetimg / offsetimg.max()
        self.baseimg = baseimg
        self.offsetimg = offsetimg
        self.gradx = np.zeros((self.ny, self.nx))
        self.grady = np.zeros((self.ny, self.nx))
        indices_list = [(ii, jj) for ii in range(self.nx) for jj in range(self.ny)]
        tasks = [self.run_in_process_pool(self._find_dots_correlate, i) for i in indices_list]
        asyncio.run(self.run_tasks(tasks))
        self.gradxy[0] = self.gradx
        self.gradxy[1] = self.grady

    def _find_dots_correlate(self, indices):
        """finds new spots using each section correlated with the center"""
        ix = indices[1]
        iy = indices[0]
        hsp = self.hsp
        r = int(np.floor(self.diameter / 2.))
        bot_base = self.y_center_base - r * self.px_spacing
        left_base = self.x_center_base - r * self.px_spacing
        vert_base = int(bot_base + iy * self.px_spacing)
        horiz_base = int(left_base + ix * self.px_spacing)
        bot_offset = self.y_center_offset - r * self.px_spacing
        left_offset = self.x_center_offset - r * self.px_spacing
        vert_offset = int(bot_offset + iy * self.px_spacing)
        horiz_offset = int(left_offset + ix * self.px_spacing)
        sec = self.offsetimg[(vert_offset - hsp):(vert_offset + hsp), (horiz_offset - hsp):(horiz_offset + hsp)]
        secbase = self.baseimg[(vert_base - hsp):(vert_base + hsp), (horiz_base - hsp):(horiz_base + hsp)]
        # indsec = N.where(sec == np.amax(sec))
        # indsecbase = N.where(sec == np.amax(secbase))
        sec = self._sub_bg(sec)
        secbase = self._sub_bg(secbase)
        seccorr = corr(1.0 * secbase, 1.0 * sec[::-1, ::-1], mode='full')
        # secbasecorr = corr(1.0*secbase, 1.0*secbase[::-1,::-1], mode='full')
        py, px = self._parabolicfit(seccorr)
        # self.CorrCenter = np.unravel_index(secbasecorr.argmax(), secbasecorr.shape)
        self.gradx[iy, ix] = self.CorrCenter[1] - px
        self.grady[iy, ix] = self.CorrCenter[0] - py
        self.im[0, iy * 2 * self.hsp: iy * 2 * self.hsp + 2 * self.hsp,
        ix * 2 * self.hsp: ix * 2 * self.hsp + 2 * self.hsp] = secbase
        self.im[1, iy * 2 * self.hsp: iy * 2 * self.hsp + 2 * self.hsp,
        ix * 2 * self.hsp: ix * 2 * self.hsp + 2 * self.hsp] = sec

    def _sub_bg(self, img):
        thresh = threshold_otsu(img)
        binary = img > thresh
        imgsub = (img - thresh) * binary
        return imgsub

    def _gaussian2d(self, x, y, x0, y0, sigmax, sigmay, a):
        return a * np.exp(-((x - x0) / sigmax) ** 2 - ((y - y0) / sigmay) ** 2)

    def _parabolicfit(self, sec):
        try:
            MaxIntLoc = np.unravel_index(sec.argmax(), sec.shape)
            secsmall = sec[(MaxIntLoc[0] - 1):(MaxIntLoc[0] + 2), (MaxIntLoc[1] - 1):(MaxIntLoc[1] + 2)]
            gradx = MaxIntLoc[1] + 0.5 * (1.0 * secsmall[1, 0] - 1.0 * secsmall[1, 2]) / (
                    1.0 * secsmall[1, 0] + 1.0 * secsmall[1, 2] - 2.0 * secsmall[1, 1])
            grady = MaxIntLoc[0] + 0.5 * (1.0 * secsmall[0, 1] - 1.0 * secsmall[2, 1]) / (
                    1.0 * secsmall[0, 1] + 1.0 * secsmall[2, 1] - 2.0 * secsmall[1, 1])
        except:  # IndexError
            gradx = self.CorrCenter[1]
            grady = self.CorrCenter[0]
        return grady, gradx

    def _disc(self, radius, size):
        x, y = size
        xv, yv = self._cartesian_grid(x, y)
        return xv * xv + yv * yv <= radius * radius

    def _polar_grid(self, x, y):
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    def _cartesian_grid(self, nx, ny):
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        return np.meshgrid(y, x)

    def _zernike_j_nm(self, j):
        if j < 1:
            raise ValueError("j must be a positive integer")
        n = 0
        while j > n:
            n += 1
            j -= n
        m = -2 * j + n
        if n % 2 == 0:
            m = -m
        return n, m

    def _zernike(self, n, m, rho, phi):
        if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
            raise ValueError("n and m are not valid Zernike indices")
        if m < 0:
            return ((-1) ** ((n - abs(m)) / 2)) * self._zernike(n, -m, rho, phi)
        # Compute the polynomial.
        kmax = int((n - abs(m)) / 2)
        summation = 0
        for k in range(kmax + 1):
            summation += ((-1) ** k * factorial(n - k) /
                          (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                           factorial(0.5 * (n - abs(m)) - k)) *
                          rho ** (n - 2 * k))
        return summation * np.cos(m * phi)

    def _zernike_polynomials(self, nz=58, size=[64, 64]):
        # Compute the Zernike polynomials on a grid.
        x, y = size
        xv, yv = self._cartesian_grid(x, y)
        rho, phi = self._polar_grid(xv, yv)
        phi = np.pi / 2 - phi
        phi = np.mod(phi, 2 * np.pi)
        zernike = np.zeros((nz, x, y))
        msk = self._disc(1, size)
        for j in range(nz):
            n, m = self._zernike_j_nm(j + 3)
            zernike[j, :, :] = msk * self._zernike(n, m, rho, phi)
        return self._gs_orthogonalisation(zernike)

    def _gs_orthogonalisation(self, arrays, check=False):
        # Gram-Schmidt orthogonalisation
        nz, nx, ny = arrays.shape
        ortharray = np.zeros((nz, nx, ny))
        ortharray[0] = arrays[0] / np.linalg.norm(arrays[0])
        for ii in range(nz - 1):
            ii = ii + 1
            ortharray[ii] = arrays[ii]
            for jj in range(ii - 1):
                ortharray[ii] = ortharray[ii] - (np.conjugate(ortharray[jj]).T * ortharray[ii]) * ortharray[jj]
            ortharray[ii] = ortharray[ii] / np.linalg.norm(ortharray[ii])
        if check:
            cmat = np.zeros((nz, nz))
            for ii in range(nz):
                for jj in range(nz):
                    w1 = ortharray[ii]
                    w2 = ortharray[jj]
                    cmat[ii, jj] = (w1 * w2.conj()).sum() / (w2 * w2.conj()).sum()
            plt.imshow(cmat)
            plt.show()
        return ortharray

    def _zernike_derivative(self, nz=58, size=[64, 64]):
        # Compute the Zernike polynomials on a grid.
        x, y = size
        xv, yv = self._cartesian_grid(x, y)
        rho, phi = self._polar_grid(xv, yv)
        phi = np.pi / 2 - phi
        phi = np.mod(phi, 2 * np.pi)
        zernike = np.zeros((nz, x, y))
        msk = self._disc(1, size)
        for j in range(nz):
            n, m = self._zernike_j_nm(j + 3)
            zernike[j, :, :] = self._zernike(n, m, rho, phi)
        dx = 2.0 / x
        dy = 2.0 / y
        d_dx = FinDiff(0, dx, 1)
        d_dy = FinDiff(1, dy, 1)
        dzarr = np.zeros((nz * 2, x, y))
        for ii in range(nz):
            dzarr[ii * 2] = d_dx(zernike[ii]) * msk
            dzarr[ii * 2 + 1] = d_dy(zernike[ii]) * msk
        return dzarr

    def _Z(self, dz):
        nz, nx, ny = dz.shape
        Z = np.zeros((int(nz / 2), nx * ny * 2))
        for ii in range(int(nz / 2)):
            Z[:nx * ny, ii] = dz[2 * ii].flatten()
            Z[nx * ny:, ii] = dz[2 * ii + 1].flatten()
        return Z

    def _zernike_coefficients(self, slopes, Z):
        u, s, vh = np.linalg.svd(Z, full_matrices=True)
        Zplus = np.dot(vh.T, np.dot(np.diag(np.diag(1 / s)), u.T))
        return np.dot(Zplus, slopes)

    async def run_in_process_pool(self, func, *args):
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=4)
        result = await loop.run_in_executor(executor, func, *args)
        return result

    async def run_tasks(self, tasks):
        results = await asyncio.gather(*tasks)
        return results
