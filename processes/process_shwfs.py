import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
from scipy.signal import fftconvolve as corr
from scipy.special import factorial
from skimage.filters import threshold_otsu

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
pi = np.pi

control_matrix_wavefront = tf.imread(
    r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_wavefront_20230413_1125.tif')
control_matrix_zonal = tf.imread(r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_20230413_1125.tif')
control_matrix_modal = tf.imread(r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_modal_20230407_2027.tif')
initial_flat = r'C:\Users\ruizhe.lin\Documents\data\dm_files\20230411_2047_flatfile.xlsx'


class WavefrontSensing:

    def __init__(self):
        self._n_actuators = 97
        self._n_lenslets_x = 19
        self._n_lenslets_y = 18
        self._n_lenslets = self._n_lenslets_x * self._n_lenslets_y
        self.x_center_base = 983
        self.y_center_base = 1081
        self.x_center_offset = 983
        self.y_center_offset = 1081
        self._lenslet_spacing = 61  # spacing between each lenslet
        self.hsp = 24  # size of subimage is 2 * hsp
        self.calfactor = (.0065 / 5.2) * 150  # pixel size * focalLength * pitch
        self.md = 'centerofmass'  # 'correlation'
        self.mag = 2
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)
        self.base = np.array([])
        self.offset = np.array([])
        self.wf = np.array([])
        self.amp = 0.1 * 2
        self._n_zernikes = 60
        self._az = None
        self.zernike = self.get_zernike_polynomials(nz=self._n_zernikes, size=[self._n_lenslets_y, self._n_lenslets_x])
        self.zslopes = self._zernike_slopes(nz=self._n_zernikes, size=[self._n_lenslets_y, self._n_lenslets_x])
        self._correction = []
        self._temp_cmd = []
        self._dm_cmd = [[0.] * 97]
        self._dm_cmd.append(self._read_cmd(initial_flat))
        self.current_cmd = 1

    def update_parameters(self, parameters):
        self.x_center_base = parameters[0]
        self.y_center_base = parameters[1]
        self.x_center_offset = parameters[2]
        self.y_center_offset = parameters[3]
        self._n_lenslets_x = parameters[4]
        self._n_lenslets_y = parameters[5]
        self._lenslet_spacing = parameters[6]
        self.hsp = parameters[7]
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)

    def get_zernike_polynomials(self, nz=60, size=None):
        # Compute the Zernike polynomials on a grid.
        if size is None:
            size = [16, 16]
        y, x = size
        yv, xv = self._cartesian_grid(x, y)
        rho, phi = self._polar_grid(xv, yv)
        phi = np.pi / 2 - phi
        phi = np.mod(phi, 2 * np.pi)
        zernike = np.zeros((nz, y, x))
        msk = self._elliptical_mask((y / 2, x / 2), (y, x))
        for j in range(nz):
            n, m = self._zernike_j_nm(j + 1)
            zernike[j, :, :] = msk * self._zernike(n, m, rho, phi)
        return self._gs_orthogonalisation(zernike)

    def wavefront_reconstruction(self, base, offset, rt=False):
        gradx, grady = self._get_gradient_xy(base, offset)
        gradx = np.pad(gradx, ((1, 1), (1, 1)), 'constant')
        grady = np.pad(grady, ((1, 1), (1, 1)), 'constant')
        extx, exty = self._hudgins_extend_mask(gradx, grady)
        phi = self._reconstruction_hudgins(extx, exty)
        phicorr = self._remove_global_waffle(phi)
        msk = self._elliptical_mask((self._n_lenslets_y / 2, self._n_lenslets_x / 2),
                                    (self._n_lenslets_y + 2, self._n_lenslets_x + 2))
        phicorr = phicorr * msk
        self.wf = phicorr[1:1 + self._n_lenslets_y, 1:1 + self._n_lenslets_x]
        if rt:
            return self.wf

    def _get_gradient_xy(self, base, offset, md='correlation'):
        """ Determines Gradients by Correlating each section with its base reference section"""
        nx = self._n_lenslets_x
        ny = self._n_lenslets_y
        hsp = self.hsp
        rx = int(nx / 2.)
        ry = int(ny / 2.)
        bot_base = self.y_center_base - ry * self._lenslet_spacing
        left_base = self.x_center_base - rx * self._lenslet_spacing
        bot_offset = self.y_center_offset - ry * self._lenslet_spacing
        left_offset = self.x_center_offset - rx * self._lenslet_spacing
        base = self._sub_back(base, 1.)
        offset = self._sub_back(offset, 1.)
        self.im = np.zeros((2, 2 * self.hsp * ny, 2 * self.hsp * nx))
        gradx = np.zeros((ny, nx))
        grady = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                vert_base = int(bot_base + iy * self._lenslet_spacing)
                horiz_base = int(left_base + ix * self._lenslet_spacing)
                vert_offset = int(bot_offset + iy * self._lenslet_spacing)
                horiz_offset = int(left_offset + ix * self._lenslet_spacing)
                secbase = base[(vert_base - hsp):(vert_base + hsp), (horiz_base - hsp):(horiz_base + hsp)]
                sec = offset[(vert_offset - hsp):(vert_offset + hsp), (horiz_offset - hsp):(horiz_offset + hsp)]
                self.im[0, iy * 2 * hsp: (iy + 1) * 2 * hsp, ix * 2 * hsp: (ix + 1) * 2 * hsp] = secbase
                self.im[1, iy * 2 * hsp: (iy + 1) * 2 * hsp, ix * 2 * hsp: (ix + 1) * 2 * hsp] = sec
                if md == 'correlation':
                    seccorr = corr(1.0 * secbase, 1.0 * sec[::-1, ::-1], mode='full')
                    py, px = self._parabolicfit(seccorr)
                    gradx[iy, ix] = (self.CorrCenter[1] - px) * self.calfactor
                    grady[iy, ix] = (self.CorrCenter[0] - py) * self.calfactor
                elif md == 'centerofmass':
                    sy, sx = self._center_of_mass(secbase)
                    py, px = self._center_of_mass(sec)
                    gradx[iy, ix] = (px - sx) * self.calfactor
                    grady[iy, ix] = (py - sy) * self.calfactor
        return gradx, grady

    @staticmethod
    def _hudgins_extend_mask(gradx, grady):
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

    @staticmethod
    def _reconstruction_hudgins(gradx, grady):
        """ wavefront reconstruction from gradients Hudgins Geometry, Poyneer 2002 """
        sx = fft2(gradx)
        sy = fft2(grady)
        ny, nx = gradx.shape
        ky, kx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        numx = (np.exp(-2j * pi * kx / nx) - 1)
        numy = (np.exp(-2j * pi * ky / ny) - 1)
        den = 4 * (np.sin(pi * kx / nx) ** 2 + np.sin(pi * ky / ny) ** 2)
        sw = (numx * sx + numy * sy) / den
        sw[0, 0] = 0.0
        return (ifft2(sw)).real

    @staticmethod
    def _remove_global_waffle(phi):
        ny, nx = phi.shape
        wmode = np.zeros((ny, nx))
        constant_num = 0
        constant_den = 0
        # a waffle-mode vector of +-1 for a given pixel of the Wavefront
        for x in range(nx):
            for y in range(ny):
                if (x + y) / 2 - np.round((x + y) / 2) == 0:
                    wmode[y, x] = 1
                else:
                    wmode[y, x] = -1
        for i in range(ny):
            for k in range(nx):
                temp = phi[i, k] * wmode[i, k]
                temp2 = wmode[i, k] * wmode[i, k]
                constant_num = constant_num + temp
                constant_den = constant_den + temp2
        constant = constant_num / constant_den
        return phi - constant * wmode

    def wavefront_decomposition(self, wf):
        self._az = np.zeros(self._n_zernikes)
        for i in range(self._n_zernikes):
            wz = self.zernike[i]
            self._az[i] = (wf * wz.conj()).sum() / (wz * wz.conj()).sum()

    def wavefront_recomposition(self, size=None):
        if size is None:
            ny = self._n_lenslets_y
            nx = self._n_lenslets_x
        else:
            ny, nx = size
        self.wf = np.zeros((ny, nx))
        for i in range(self._n_zernikes):
            self.wf += self._az[i] * self.zernike[i]

    def _parabolicfit(self, sec):
        try:
            MaxIntLoc = np.unravel_index(sec.argmax(), sec.shape)
            secsmall = sec[(MaxIntLoc[0] - 1):(MaxIntLoc[0] + 2), (MaxIntLoc[1] - 1):(MaxIntLoc[1] + 2)]
            gradx = MaxIntLoc[1] + 0.5 * (1.0 * secsmall[1, 0] - 1.0 * secsmall[1, 2]) / (
                    1.0 * secsmall[1, 0] + 1.0 * secsmall[1, 2] - 2.0 * secsmall[1, 1])
            grady = MaxIntLoc[0] + 0.5 * (1.0 * secsmall[0, 1] - 1.0 * secsmall[2, 1]) / (
                    1.0 * secsmall[0, 1] + 1.0 * secsmall[2, 1] - 2.0 * secsmall[1, 1])
        except:  # IndexError
            gradx = self.CorrCenter[0]
            grady = self.CorrCenter[1]
        return grady, gradx

    @staticmethod
    def _sub_back(img, factor):
        thresh = factor * threshold_otsu(img)
        binary = img > thresh
        return (img - thresh) * binary

    def generate_influence_matrix(self, data_folder, method='wavefront'):
        if method == 'wavefront':
            _influence_matrix = np.zeros((self._n_lenslets, self._n_actuators))
        elif method == 'zonal':
            _influence_matrix = np.zeros((2 * self._n_lenslets, self._n_actuators))
        elif method == 'modal':
            _influence_matrix = np.zeros((self._n_zernikes, self._n_actuators))
        else:
            raise ValueError("Invalid method")
        _msk = self._elliptical_mask((self._n_lenslets_y / 2, self._n_lenslets_x / 2),
                                     (self._n_lenslets_y, self._n_lenslets_x))
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif"):
                ind = int(filename.split("_")[3])
                print(filename.split("_")[3])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                n, x, y = data_stack.shape
                if n != 4:
                    raise "The image number has to be 4"
                if method == 'wavefront':
                    wfp = self.wavefront_reconstruction(data_stack[0], data_stack[1], rt=True)
                    # wfn = self.wavefront_reconstruction(data_stack[2], data_stack[3], rt=True)
                    # _influence_matrix[:, ind] = ((wfp - wfn) / self.amp).reshape(self._n_lenslets)
                    msk = (wfp != 0.0).astype(np.float32)
                    mn = wfp.sum() / msk.sum()
                    wfp = msk * (wfp - mn)
                    _influence_matrix[:, ind] = (wfp / (0.5 * self.amp)).reshape(self._n_lenslets)
                else:
                    gdxp, gdyp = self._get_gradient_xy(data_stack[0], data_stack[1])
                    gdxn, gdyn = self._get_gradient_xy(data_stack[2], data_stack[3])
                    if method == 'zonal':
                        _influence_matrix[:self._n_lenslets, ind] = ((gdxp - gdxn) / self.amp).reshape(self._n_lenslets)
                        _influence_matrix[self._n_lenslets:, ind] = ((gdyp - gdyn) / self.amp).reshape(self._n_lenslets)
                    if method == 'modal':
                        a1 = self._zernike_coefficients(np.concatenate((gdxp.flatten(), gdyp.flatten())), self.zslopes)
                        a2 = self._zernike_coefficients(np.concatenate((gdxn.flatten(), gdyn.flatten())), self.zslopes)
                        _influence_matrix[:, ind] = ((a1 - a2) / self.amp).flatten()
        return _influence_matrix

    def get_control_matrix(self, influence_matrix, n=81):
        return self._pseudo_inverse(influence_matrix, n=n)

    def get_correction(self, measurement, method='wavefront'):
        if method == 'wavefront':
            mwf = self.wavefront_reconstruction(self.base, measurement, rt=True)
            self._correction.append(
                list(0.5 * self.amp * np.dot(control_matrix_wavefront, mwf.reshape(self._n_lenslets))))
        else:
            gradx, grady = self._get_gradient_xy(self.base, measurement)
            _measurement = np.concatenate((gradx.reshape(self._n_lenslets), grady.reshape(self._n_lenslets)))
            if method == 'zonal':
                self._correction.append(list(np.dot(control_matrix_zonal, _measurement)))
            elif method == 'modal':
                a = self._zernike_coefficients(_measurement, self.zslopes)
                self._correction.append(list(np.dot(control_matrix_modal, a)))
            else:
                raise ValueError("Invalid method")

    def correct_cmd(self):
        _c = np.asarray(self._dm_cmd[self.current_cmd]) + np.asarray(self._correction[-1])
        self._dm_cmd.append(list(_c))

    def _read_cmd(self, fnd):
        df = pd.read_excel(fnd)
        return df['Push'].tolist()

    def _write_cmd(self, path, t, flatfile=False):
        if flatfile:
            filename = t + '_flat_file.csv'
            df = pd.DataFrame(self._dm_cmd[-1], index=np.arange(97), columns=['Push'])
            df.to_excel(os.path.join(path, filename), index_label='Actuator')
        else:
            filename = t + '_cmd_file.csv'
            data = {f'cmd{i}': cmd for i, cmd in enumerate(self._dm_cmd)}
            writer = pd.ExcelWriter(os.path.join(path, filename), engine='xlsxwriter')
            workbook = writer.book
            for sheet_name, list_data in data.items():
                df = pd.DataFrame(list_data, index=np.arange(97), columns=['Push'])
                df.to_excel(writer, sheet_name=sheet_name, index_label='Actuator')
            writer.close()

    @staticmethod
    def _elliptical_mask(radius, size):
        coord_x = np.arange(0.5, size[0], 1.0)
        coord_y = np.arange(0.5, size[1], 1.0)
        y, x = np.meshgrid(coord_y, coord_x)
        x -= size[0] / 2.
        y -= size[1] / 2.
        return (x * x / (radius[0] * radius[0])) + (y * y / (radius[1] * radius[1])) <= 1

    @staticmethod
    def _circular_mask(radius, size):
        coords = np.arange(0.5, size, 1.0)
        x, y = np.meshgrid(coords, coords)
        x -= size / 2.
        y -= size / 2.
        return x * x + y * y <= radius * radius

    @staticmethod
    def _pseudo_inverse(A, n=52):
        U, s, Vt = np.linalg.svd(A)
        s_inv = np.zeros_like(A.T)
        if n is None:
            s_inv[:min(A.shape), :min(A.shape)] = np.diag(1 / s[:min(A.shape)])
        else:
            s_inv[:n, :n] = np.diag(1 / s[:n])
        return Vt.T @ s_inv @ U.T

    @staticmethod
    def _center_of_mass(image):
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

    @staticmethod
    def _polar_grid(x, y):
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)

    @staticmethod
    def _cartesian_grid(nx, ny):
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        return np.meshgrid(y, x, indexing='ij')

    @staticmethod
    def _zernike_j_nm(j):
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

    @staticmethod
    def _zernike(n, m, rho, phi):
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

    @staticmethod
    def _zernike_derivatives(n, m, rho, phi):
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

    def _zernike_slopes(self, nz=58, size=None):
        # Compute the Zernike polynomials on a grid.
        if size is None:
            size = [64, 64]
        x, y = size
        xv, yv = self._cartesian_grid(x, y)
        rho, phi = self._polar_grid(xv, yv)
        phi = np.pi / 2 - phi
        phi = np.mod(phi, 2 * np.pi)
        msk = 1
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

    @staticmethod
    def _gs_orthogonalisation(arrays):
        nz, ny, nx = arrays.shape
        ortharray = np.zeros((nz, ny, nx))
        ortharray[0] = arrays[0] / np.linalg.norm(arrays[0])
        for ii in range(1, nz):
            ortharray[ii] = arrays[ii]
            for jj in range(ii):
                inner_product = np.einsum('ij,ij->', np.conjugate(ortharray[jj]), ortharray[ii])
                ortharray[ii] = ortharray[ii] - inner_product * ortharray[jj]
            norm = np.linalg.norm(ortharray[ii], axis=(0, 1))
            ortharray[ii] = ortharray[ii] / norm
        return ortharray
