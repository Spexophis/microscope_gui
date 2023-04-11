import asyncio
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile as tf
from scipy.ndimage import center_of_mass as com
from scipy.signal import fftconvolve as corr
from scipy.special import factorial
from skimage.filters import threshold_otsu

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
pi = np.pi

control_matrix_zonal = tf.imread(r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_zonal_20230411_1441.tif')
control_matrix_modal = tf.imread(r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_modal_20230407_2027.tif')


# flat_start


class WavefrontSensing:

    def __init__(self):
        self.radius = 9  # 1/2 the total number of lenslets in linear direction
        self.diameter = 18  # total number of lenslets in linear direction
        self.x_center_base = 1012
        self.y_center_base = 1079
        self.x_center_offset = 1012
        self.y_center_offset = 1079
        self.px_spacing = 60  # spacing between each lenslet
        self.hsp = 32  # size of subimage is 2*hsp
        self.calfactor = (.0065 / 5.2) * 150  # pixel size * focalLength * pitch
        # set up seccorr center
        self.md = 'correlation'  # 'centerofmass'
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)
        self.base = np.array([])
        self.offset = np.array([])
        self.wf = np.array([])
        self.amp = 0.1 * 2
        self._n_zernikes = 58
        self._az = None
        self.zernike = self._zernike_polynomials(nz=self._n_zernikes, size=[self.diameter, self.diameter])
        self.zslopes = self._zernike_slopes(nz=self._n_zernikes, size=[self.diameter, self.diameter])
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

    def _generate_influence_matrix(self, data_folder, method='zonal'):
        self._n_actuators = 97
        self._n_lenslets = self.diameter * self.diameter
        if method == 'zonal':
            _influence_matrix = np.zeros((2 * self._n_lenslets, self._n_actuators))
        elif method == 'modal':
            _influence_matrix = np.zeros((self._n_zernikes, self._n_actuators))
        else:
            raise ValueError("Invalid method")
        _msk = self._circular_mask(self.diameter / 2, self.diameter)
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif"):
                ind = int(filename.split("_")[3])
                print(filename.split("_")[3])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                n, x, y = data_stack.shape
                if n != 4:
                    raise "The image number has to be 4"
                gdxp, gdyp = self._get_gradient_xy(data_stack[0], data_stack[1])
                gdxn, gdyn = self._get_gradient_xy(data_stack[2], data_stack[3])
                if method == 'zonal':
                    _influence_matrix[:self._n_lenslets, ind] = ((gdxp * _msk - gdxn * _msk) / self.amp).flatten()
                    _influence_matrix[self._n_lenslets:, ind] = ((gdyp * _msk - gdyn * _msk) / self.amp).flatten()
                if method == 'modal':
                    a1 = self._zernike_coefficients(np.concatenate((gdxp.flatten(), gdyp.flatten())), self.zslopes)
                    a2 = self._zernike_coefficients(np.concatenate((gdxn.flatten(), gdyn.flatten())), self.zslopes)
                    _influence_matrix[:, ind] = ((a1 - a2) / self.amp).flatten()
        return _influence_matrix

    def _get_control_matrix(self, influence_matrix):
        return self._pseudo_inverse(influence_matrix)

    def _get_correction(self, measurement, method='zonal'):
        gradx, grady = self._get_gradient_xy(self.base, measurement)
        _measurement = np.concatenate((gradx.flatten(), grady.flatten()))
        if method == 'zonal':
            self._correction.append(np.matmul(control_matrix_zonal, _measurement))
        elif method == 'modal':
            a = self._zernike_coefficients(_measurement, self.zslopes)
            self._correction.append(np.matmul(control_matrix_modal, a))
        else:
            raise ValueError("Invalid method")
        return self._correction[-1]

    def _correct_cmd(self):
        _c = self._dm_cmd[-1] + self._correction
        self._dm_cmd.append(_c)

    def _get_gradient_xy(self, base, offset, cocurrent=False):
        """ Determines Gradients by Correlating each section with its base reference section"""
        self.nx = self.diameter
        self.ny = self.diameter
        self.im = np.zeros((2, 2 * self.hsp * self.diameter, 2 * self.hsp * self.diameter))
        gradx = np.zeros((self.ny, self.nx))
        grady = np.zeros((self.ny, self.nx))
        self.baseimg = base
        self.offsetimg = offset
        indices_list = [(ii, jj) for ii in range(self.nx) for jj in range(self.ny)]
        if cocurrent:
            tasks = [self.run_in_process_pool(self._find_dot_center, i) for i in indices_list]
            results = asyncio.run(self.run_tasks(tasks))
        else:
            for indices in indices_list:
                gradx[indices[0], indices[1]], grady[indices[0], indices[1]], ind = self._find_dot_center(indices)
        return gradx, grady

    def _find_dot_center(self, indices):
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
        # self._detect_gaussian_object(sec, sigma=4)
        # sec = self._sub_bg(sec)
        # secbase = self._sub_bg(secbase)
        if self.md == 'correlation':
            seccorr = corr(1.0 * secbase, 1.0 * sec[::-1, ::-1], mode='full')
            py, px = self._parabolicfit(seccorr)
            gdx = self.CorrCenter[1] - px
            gdy = self.CorrCenter[0] - py
        elif self.md == 'centerofmass':
            py, px = com(sec)
            sy, sx = com(secbase)
            gdx = px - sx
            gdy = py - sy
        return gdx, gdy, indices

    def _detect_gaussian_object(self, image, sigma):
        image = image / image.sum()
        # Apply the Difference of Gaussians (DoG) filter
        filtered = ndi.gaussian_filter(image, sigma=sigma) - ndi.gaussian_filter(image, sigma=sigma * 1.6)
        # Find the local maxima with a higher threshold
        threshold = 0.2 * np.max(filtered)
        coordinates = np.transpose(np.nonzero(ndi.maximum_filter(filtered, size=3) == filtered))
        coordinates = coordinates[filtered[coordinates[:, 0], coordinates[:, 1]] > threshold]
        # Filter out small and noisy points
        size_threshold = sigma * 16
        sizes = []
        for y, x in coordinates:
            y0 = max(y - image.shape[0] // 2, 0)
            y1 = min(y + image.shape[0] // 2 + 1, image.shape[0])
            x0 = max(x - image.shape[1] // 2, 0)
            x1 = min(x + image.shape[1] // 2 + 1, image.shape[1])
            yy, xx = np.mgrid[y0:y1, x0:x1]
            data = image[y0:y1, x0:x1]
            fit = np.polyfit(np.arange(data.size), data.ravel(), deg=2)
            sigma = np.sqrt(-1 / (2 * fit[0]))
            size = np.sqrt(2 * np.pi) * sigma
            sizes.append(size)
        sizes = np.array(sizes)
        coordinates = coordinates[sizes > size_threshold]
        # return coordinates
        if coordinates.shape == (1, 2):
            return True
            # print(coordinates)
        else:
            return False
            # print("No Gaussian-like points found in the image")

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

    def _center_of_mass(self, image):
        height, width = image.shape
        # row_indices, col_indices = np.indices((height, width))
        # total_mass = np.sum(image)
        # row_mass = np.sum(row_indices * image) / total_mass
        # col_mass = np.sum(col_indices * image) / total_mass
        row_indices = np.arange(0, height)[:, np.newaxis]
        col_indices = np.arange(0, width)
        total_mass = np.sum(image)
        row_mass = np.sum(image * row_indices) / total_mass
        col_mass = np.sum(image * col_indices) / total_mass
        return row_mass, col_mass

    def _disc(self, radius, size):
        x, y = size
        xv, yv = self._cartesian_grid(x, y)
        return xv * xv + yv * yv <= radius * radius

    def _polar_grid(self, x, y):
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    def _cartesian_grid(self, nx, ny):
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        return np.meshgrid(x, y)

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
        kmax = int((n - abs(m)) / 2)
        R = 0
        for k in range(kmax + 1):
            R += ((-1) ** k * factorial(n - k) /
                  (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                   factorial(0.5 * (n - abs(m)) - k)) *
                  rho ** (n - 2 * k))
        if m >= 0:
            O = np.cos(m * phi)
        if m < 0:
            O = np.sin(np.abs(m) * phi)
        return R * O

    def _zernike_derivatives(self, n, m, rho, phi):
        if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
            raise ValueError("n and m are not valid Zernike indices")
        kmax = int((n - abs(m)) / 2)
        R = 0
        dR = 0
        for k in range(kmax + 1):
            R += ((-1) ** k * factorial(n - k) /
                  (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                   factorial(0.5 * (n - abs(m)) - k)) *
                  rho ** (n - 2 * k))
            dR += ((-1) ** k * factorial(n - k) /
                   (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                    factorial(0.5 * (n - abs(m)) - k)) *
                   (n - 2 * k) * rho ** (n - 2 * k - 1))
        if m >= 0:
            O = np.cos(m * phi)
            dO = m * np.sin(m * phi)
        if m < 0:
            O = np.sin(np.abs(m) * phi)
            dO = - np.abs(m) * np.sin(np.abs(m) * phi)
        dx = dR * O * np.cos(phi) - (R / rho) * dO * np.sin(phi)
        dy = dR * O * np.sin(phi) + (R / rho) * dO * np.cos(phi)
        return dx, dy

    def _zernike_polynomials(self, nz=58, size=[64, 64]):
        # Compute the Zernike polynomials on a grid.
        x, y = size
        xv, yv = self._cartesian_grid(x, y)
        rho, phi = self._polar_grid(xv, yv)
        phi = np.pi / 2 - phi
        phi = np.mod(phi, 2 * np.pi)
        zernike = np.zeros((nz, x, y))
        msk = self._disc(1.05, size)
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

    def _zernike_slopes(self, nz=58, size=None):
        # Compute the Zernike polynomials on a grid.
        if size is None:
            size = [64, 64]
        x, y = size
        xv, yv = self._cartesian_grid(x, y)
        rho, phi = self._polar_grid(xv, yv)
        phi = np.pi / 2 - phi
        phi = np.mod(phi, 2 * np.pi)
        msk = self._disc(1.05, size)
        zs = np.zeros((2 * x * y, nz))
        for j in range(nz):
            n, m = self._zernike_j_nm(j + 3)
            zdx, zdy = msk * self._zernike_derivatives(n, m, rho, phi)
            zs[:x * y, j] = zdx.flatten()
            zs[x * y:, j] = zdy.flatten()
        return zs

    def _zernike_coefficients(self, gradxy, gradz):
        zplus = self._pseudo_inverse(gradz)
        return np.matmul(zplus, gradxy)

    def _pseudo_inverse(self, A):
        U, s, Vt = np.linalg.svd(A)
        s_inv = np.zeros_like(A.T, dtype=float)
        s_inv[:min(A.shape), :min(A.shape)] = np.diag(1 / s[:min(A.shape)])
        return Vt.T @ s_inv @ U.T

    def _wavefront_reconstruction(self, base, offset):
        gradx, grady = self._get_gradient_xy(base, offset)
        gradx = np.pad(gradx, ((1, 1), (1, 1)), 'constant')
        grady = np.pad(grady, ((1, 1), (1, 1)), 'constant')
        extx, exty = self._hudgins_extend_mask(gradx, grady)
        phi = self._reconstruction_hudgins(extx, exty)
        phi = phi * self.calfactor
        phicorr = self._remove_global_waffle(phi)
        msk = self._circular_mask(self.diameter / 2., self.diameter + 2)
        phicorr = phicorr * msk
        self.wf = phicorr[1:1 + self.diameter, 1:1 + self.diameter]

    def _hudgins_extend_mask(self, gradx, grady):
        """ extension technique Poyneer 2002 """
        nx, ny = gradx.shape
        if nx % 2 == 0:  # even
            mx = nx / 2
        else:  # odd
            mx = (nx + 1) / 2
        if ny % 2 == 0:  # even
            my = ny / 2
        else:  # odd
            my = (ny + 1) / 2
        for jj in range(int(nx)):
            for ii in range(int(my), int(ny)):
                if grady[jj, ii] == 0.0:
                    grady[jj, ii] = grady[jj, ii - 1]
            for ii in range(int(my), -1, -1):
                if grady[jj, ii] == 0.0:
                    grady[jj, ii] = grady[jj, ii + 1]
        for jj in range(int(ny)):
            for ii in range(int(mx), int(nx)):
                if gradx[ii, jj] == 0.0:
                    gradx[ii, jj] = gradx[ii - 1, jj]
            for ii in range(int(mx), -1, -1):
                if gradx[ii, jj] == 0.0:
                    gradx[ii, jj] = gradx[ii + 1, jj]
        gradxe = gradx.copy()
        gradye = grady.copy()
        gradxe[:, ny - 1] = -1.0 * gradx[:, :(ny - 1)].sum(1)
        gradye[nx - 1, :] = -1.0 * grady[:(nx - 1), :].sum(0)
        return gradxe, gradye

    def _reconstruction_hudgins(self, gradx, grady):
        """ wavefront reconstruction from gradients Hudgins Geometry, Poyneer 2002 """
        sx = fft2(gradx)
        sy = fft2(grady)
        nx, ny = gradx.shape
        k, l = np.meshgrid(np.arange(ny), np.arange(nx))
        numx = (np.exp(-2j * pi * k / nx) - 1)
        numy = (np.exp(-2j * pi * l / ny) - 1)
        den = 4 * (np.sin(pi * k / nx) ** 2 + np.sin(pi * l / ny) ** 2)
        sw = (numx * sx + numy * sy) / den
        sw[0, 0] = 0.0
        return (ifft2(sw)).real

    def _remove_global_waffle(self, phi):
        wmode = np.zeros((self.diameter + 2, self.diameter + 2))
        constant_num = 0
        constant_den = 0
        # a waffle-mode vector of +-1 for a given pixel of the Wavefront
        for x in range(self.diameter + 2):
            for y in range(self.diameter + 2):
                if (x + y) / 2 - np.round((x + y) / 2) == 0:
                    wmode[y, x] = 1
                else:
                    wmode[y, x] = -1
        for i in range(self.diameter + 2):
            for k in range(self.diameter + 2):
                temp = phi[i, k] * wmode[i, k]
                temp2 = wmode[i, k] * wmode[i, k]
                constant_num = constant_num + temp
                constant_den = constant_den + temp2
        constant = constant_num / constant_den
        return phi - constant * wmode

    def _wavefront_decomposition(self, wf):
        self._az = np.zeros(self._n_zernikes)
        for i in range(self._n_zernikes):
            wz = self.zernike[i]
            self._az[i] = (wf * wz.conj()).sum() / (wz * wz.conj()).sum()

    def _wavefront_recomposition(self, d=None):
        if d is None:
            d = self.diameter
        self.wf = np.zeros((d, d))
        for i in range(self._n_zernikes):
            self.wf += self._az[i] * self.zernike[i]

    def _circular_mask(self, radius, size, circle_centre=(0, 0), origin="middle"):
        coords = np.arange(0.5, size, 1.0)
        x, y = np.meshgrid(coords, coords)
        x -= size / 2.
        y -= size / 2.
        return x * x + y * y <= radius * radius

    async def run_in_process_pool(self, func, *args):
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=4)
        result = await loop.run_in_executor(executor, func, *args)
        return result

    async def run_tasks(self, tasks):
        results = await asyncio.gather(*tasks)
        return results
