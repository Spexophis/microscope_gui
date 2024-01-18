import os
import threading

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

control_matrix_phase = tf.imread(r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_phase_20230726.tif')
control_matrix_zonal = tf.imread(r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_zonal_20230726.tif')
control_matrix_modal = tf.imread(r'C:\Users\ruizhe.lin\Documents\data\dm_files\control_matrix_modal_20230613.tif')
initial_flat = r'C:\Users\ruizhe.lin\Documents\data\dm_files\flatfile_20230728.xlsx'


class WavefrontSensing:

    def __init__(self):
        self.n_actuators = 97
        self.n_lenslets_x = 34
        self.n_lenslets_y = 35
        self.n_lenslets = self.n_lenslets_x * self.n_lenslets_y
        self.x_center_base = 1060
        self.y_center_base = 1085
        self.x_center_offset = 1060
        self.y_center_offset = 1085
        self.lenslet_spacing = 24  # spacing between each lenslet
        self.hsp = 12  # size of subimage is 2 * hsp
        self.calfactor = (.0065 / 5.2) * 150  # pixel size * focalLength * pitch
        self.method = 'correlation'
        self.mag = 2
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)
        self.base = np.array([])
        self.offset = np.array([])
        self.wf = np.array([])
        self.recon_thread = None
        self.amp = 0.1
        self.n_zernikes = 60
        self.az = None
        self.zernike = self.get_zernike_polynomials(nz=self.n_zernikes, size=[self.n_lenslets_y, self.n_lenslets_x])
        self.zslopes = self.get_zernike_slopes(nz=self.n_zernikes, size=[self.n_lenslets_y, self.n_lenslets_x])
        self._correction = []
        self._temp_cmd = []
        self._dm_cmd = [[0.] * 97]
        self._dm_cmd.append(self.read_cmd(initial_flat))
        self.current_cmd = 1

    def update_parameters(self, parameters):
        self.x_center_base = parameters[0]
        self.y_center_base = parameters[1]
        self.x_center_offset = parameters[2]
        self.y_center_offset = parameters[3]
        self.n_lenslets_x = parameters[4]
        self.n_lenslets_y = parameters[5]
        self.n_lenslets = self.n_lenslets_x * self.n_lenslets_y
        self.lenslet_spacing = parameters[6]
        self.hsp = parameters[7]
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)
        self.zernike = self.get_zernike_polynomials(nz=self.n_zernikes, size=[self.n_lenslets_y, self.n_lenslets_x])
        self.zslopes = self.get_zernike_slopes(nz=self.n_zernikes, size=[self.n_lenslets_y, self.n_lenslets_x])

    def get_zernike_polynomials(self, nz=60, size=None):
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

    def get_zernike_derivatives(self, nz=60, size=None):
        if size is None:
            size = [16, 16]
        y, x = size
        yv, xv = self._cartesian_grid(x, y)
        rho, phi = self._polar_grid(xv, yv)
        phi = np.pi / 2 - phi
        phi = np.mod(phi, 2 * np.pi)
        zernike_derivatives = np.zeros((nz, 2, y, x))
        msk = self._elliptical_mask((y / 2, x / 2), (y, x))
        for j in range(nz):
            n, m = self._zernike_j_nm(j + 1)
            dx, dy = self._zernike_derivatives(n, m, rho, phi)
            zernike_derivatives[j, 0, :, :] = dy * msk
            zernike_derivatives[j, 1, :, :] = dx * msk
        return zernike_derivatives

    def get_zernike_slopes(self, nz=58, size=None):
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

    def run_wf_recon(self, callback=None):
        self.recon_thread = ReconstructionThread(self.wavefront_reconstruction, callback)
        self.recon_thread.start()

    def wavefront_reconstruction(self, rt=False):
        gradx, grady = self._get_gradient_xy(self.base, self.offset)
        gradx = np.pad(gradx, ((1, 1), (1, 1)), 'constant')
        grady = np.pad(grady, ((1, 1), (1, 1)), 'constant')
        extx, exty = self._hudgins_extend_mask(gradx, grady)
        phi = self._reconstruction_hudgins(extx, exty)
        phicorr = self._remove_global_waffle(phi)
        msk = self._elliptical_mask((self.n_lenslets_y / 2, self.n_lenslets_x / 2),
                                    (self.n_lenslets_y + 2, self.n_lenslets_x + 2))
        phicorr = phicorr * msk
        self.wf = phicorr[1:1 + self.n_lenslets_y, 1:1 + self.n_lenslets_x]
        if rt:
            return self.wf

    def _get_gradient_xy(self, base, offset, md='correlation'):
        """ Determines Gradients by Correlating each section with its base reference section"""
        nx = self.n_lenslets_x
        ny = self.n_lenslets_y
        hsp = self.hsp
        rx = int(nx / 2.)
        ry = int(ny / 2.)
        bot_base = self.y_center_base - ry * self.lenslet_spacing
        left_base = self.x_center_base - rx * self.lenslet_spacing
        bot_offset = self.y_center_offset - ry * self.lenslet_spacing
        left_offset = self.x_center_offset - rx * self.lenslet_spacing
        base = self._sub_back(base, 0.8)
        offset = self._sub_back(offset, 0.8)
        self.im = np.zeros((2, 2 * self.hsp * ny, 2 * self.hsp * nx))
        gradx = np.zeros((ny, nx))
        grady = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                vert_base = int(bot_base + iy * self.lenslet_spacing)
                horiz_base = int(left_base + ix * self.lenslet_spacing)
                vert_offset = int(bot_offset + iy * self.lenslet_spacing)
                horiz_offset = int(left_offset + ix * self.lenslet_spacing)
                secbase = base[(vert_base - hsp):(vert_base + hsp), (horiz_base - hsp):(horiz_base + hsp)]
                sec = offset[(vert_offset - hsp):(vert_offset + hsp), (horiz_offset - hsp):(horiz_offset + hsp)]
                self.im[0, iy * 2 * hsp: (iy + 1) * 2 * hsp, ix * 2 * hsp: (ix + 1) * 2 * hsp] = secbase
                self.im[1, iy * 2 * hsp: (iy + 1) * 2 * hsp, ix * 2 * hsp: (ix + 1) * 2 * hsp] = sec
                if md == 'correlation':
                    seccorr = corr(1.0 * secbase, 1.0 * sec[::-1, ::-1], mode='full')
                    py, px = self._parabolic_fit(seccorr)
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
        # sw = (numx * sx + numy * sy) / den
        sw = np.divide((numx * sx + numy * sy), den, where=den != 0)
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

    def run_wf_modal_recon(self, callback=None):
        self.recon_thread = ReconstructionThread(self.wavefront_modal_reconstruction, callback)
        self.recon_thread.start()

    def wavefront_modal_reconstruction(self):
        self.wavefront_decomposition()
        self.wavefront_recomposition()

    def wavefront_decomposition(self):
        self.az = np.zeros(self.n_zernikes)
        for i in range(self.n_zernikes):
            wz = self.zernike[i]
            self.az[i] = (self.wf * wz.conj()).sum() / (wz * wz.conj()).sum()

    def wavefront_recomposition(self, size=None, exclusive=True):
        if size is None:
            ny = self.n_lenslets_y
            nx = self.n_lenslets_x
        else:
            ny, nx = size
        self.wf = np.zeros((ny, nx))
        if exclusive:
            self.az[:4] = 0
        for i in range(self.n_zernikes):
            self.wf += self.az[i] * self.zernike[i]

    def _parabolic_fit(self, sec):
        try:
            init_max_loc = np.unravel_index(sec.argmax(), sec.shape)
            sec_zoom = sec[(init_max_loc[0] - 1):(init_max_loc[0] + 2), (init_max_loc[1] - 1):(init_max_loc[1] + 2)]
            gradx = init_max_loc[1] + 0.5 * (1.0 * sec_zoom[1, 0] - 1.0 * sec_zoom[1, 2]) / (
                    1.0 * sec_zoom[1, 0] + 1.0 * sec_zoom[1, 2] - 2.0 * sec_zoom[1, 1])
            grady = init_max_loc[0] + 0.5 * (1.0 * sec_zoom[0, 1] - 1.0 * sec_zoom[2, 1]) / (
                    1.0 * sec_zoom[0, 1] + 1.0 * sec_zoom[2, 1] - 2.0 * sec_zoom[1, 1])
        except:  # IndexError
            gradx = self.CorrCenter[0]
            grady = self.CorrCenter[1]
        return grady, gradx

    @staticmethod
    def _sub_back(img, factor):
        thresh = factor * threshold_otsu(img)
        binary = img > thresh
        return (img - thresh) * binary

    def generate_influence_matrix(self, data_folder, method='phase', sv=False):
        if method == 'phase':
            _influence_matrix = np.zeros((self.n_lenslets, self.n_actuators))
            wfs = np.zeros((self.n_actuators, self.n_lenslets_y, self.n_lenslets_x))
        elif method == 'zonal':
            _influence_matrix = np.zeros((2 * self.n_lenslets, self.n_actuators))
        elif method == 'modal':
            _influence_matrix = np.zeros((self.n_zernikes, self.n_actuators))
        else:
            raise ValueError("Invalid method")
        _msk = self._elliptical_mask((self.n_lenslets_y / 2, self.n_lenslets_x / 2),
                                     (self.n_lenslets_y, self.n_lenslets_x))
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif") & filename.startswith("actuator"):
                ind = int(filename.split("_")[1])
                print(filename.split("_")[1])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                n, x, y = data_stack.shape
                if n != 4:
                    raise "The image number has to be 4"
                if method == 'phase':
                    self.base = data_stack[0]
                    self.offset = data_stack[1]
                    wfp = self.wavefront_reconstruction(rt=True)
                    wfs[ind] = wfp
                    # wfn = self.wavefront_reconstruction(data_stack[2], data_stack[3], rt=True)
                    # _influence_matrix[:, ind] = ((wfp - wfn) / (2 * self.amp)).reshape(self.n_lenslets)
                    msk = (wfp != 0.0).astype(np.float32)
                    mn = wfp.sum() / msk.sum()
                    wfp = msk * (wfp - mn)
                    _influence_matrix[:, ind] = wfp.reshape(self.n_lenslets)
                else:
                    gdxp, gdyp = self._get_gradient_xy(data_stack[0], data_stack[1])
                    gdxn, gdyn = self._get_gradient_xy(data_stack[2], data_stack[3])
                    if method == 'zonal':
                        _influence_matrix[:self.n_lenslets, ind] = ((gdxp - gdxn) / (2 * self.amp)).reshape(
                            self.n_lenslets)
                        _influence_matrix[self.n_lenslets:, ind] = ((gdyp - gdyn) / (2 * self.amp)).reshape(
                            self.n_lenslets)
                    if method == 'modal':
                        a1 = self.get_zernike_coefficients(np.concatenate((gdxp.flatten(), gdyp.flatten())),
                                                           self.zslopes)
                        a2 = self.get_zernike_coefficients(np.concatenate((gdxn.flatten(), gdyn.flatten())),
                                                           self.zslopes)
                        _influence_matrix[:, ind] = ((a1 - a2) / (2 * self.amp)).flatten()
        _control_matrix = self._pseudo_inverse(_influence_matrix, n=32)
        if sv:
            tf.imwrite(os.path.join(data_folder, "influence_function.tif"), _influence_matrix)
            tf.imwrite(os.path.join(data_folder, "control_matrix.tif"), _control_matrix)
            if 'wfs' in locals():
                if isinstance(wfs, np.ndarray):
                    tf.imwrite(os.path.join(data_folder, "influence_function_images.tif"), wfs)

    def get_correction(self, method='phase'):
        if method == 'phase':
            mwf = self.wavefront_reconstruction(rt=True)
            self._correction.append(
                list(self.amp * np.dot(control_matrix_phase, -mwf.reshape(self.n_lenslets))))
        else:
            gradx, grady = self._get_gradient_xy(self.base, self.offset)
            _measurement = np.concatenate((gradx.reshape(self.n_lenslets), grady.reshape(self.n_lenslets)))
            if method == 'zonal':
                self._correction.append(list(np.dot(control_matrix_zonal, -_measurement)))
            elif method == 'modal':
                a = self.get_zernike_coefficients(-_measurement, self.zslopes)
                self._correction.append(list(np.dot(control_matrix_modal, a)))
            else:
                raise ValueError("Invalid method")
        _c = self.cmd_add(self._dm_cmd[self.current_cmd], self._correction[-1])
        self._dm_cmd.append(_c)

    def get_zernike_coefficients(self, gradxy, gradz):
        zplus = self._pseudo_inverse(gradz, n=32)
        return np.matmul(zplus, gradxy)

    def get_zernike_cmd(self, j, a):
        zerphs = a * self.zernike[j]
        return list(np.dot(control_matrix_phase, zerphs.reshape(self.n_lenslets)))

    def cmd_add(self, cmd_0, cmd_1):
        return list(np.asarray(cmd_0) + np.asarray(cmd_1))

    def read_cmd(self, fnd):
        df = pd.read_excel(fnd)
        return df['Push'].tolist()

    def write_cmd(self, path, t, flatfile=False):
        if flatfile:
            filename = t + '_flat_file.xlsx'
            df = pd.DataFrame(self._dm_cmd[-1], index=np.arange(97), columns=['Push'])
            df.to_excel(os.path.join(path, filename), index_label='Actuator')
        else:
            filename = t + '_cmd_file.xlsx'
            data = {f'cmd{i}': cmd for i, cmd in enumerate(self._dm_cmd)}
            with pd.ExcelWriter(os.path.join(path, filename), engine='xlsxwriter') as writer:
                for sheet_name, list_data in data.items():
                    df = pd.DataFrame(list_data, index=np.arange(97), columns=['Push'])
                    df.to_excel(writer, sheet_name=sheet_name, index_label='Actuator')

    def save_sensorless_results(self, fd, a, v, p):
        df1 = pd.DataFrame(v, index=a, columns=['Values'])
        df2 = pd.DataFrame(p, index=np.arange(self.n_zernikes), columns=['Amplitudes'])
        with pd.ExcelWriter(fd, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Metric Values')
            df2.to_excel(writer, sheet_name='Peaks')

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
    def _pseudo_inverse(A, n=32):
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
        n = int((-1. + np.sqrt(8 * (j - 1) + 1)) / 2.)
        p = (j - (n * (n + 1)) / 2.)
        k = n % 2
        m = int((p + k) / 2.) * 2 - k
        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1
            m *= s
        return n, m

    @staticmethod
    def _zernike(n, m, rho, phi):
        if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
            raise ValueError("n and m are not valid Zernike indices")
        kmax = int((n - abs(m)) / 2)
        _R = 0
        _O = 0
        _C = 0
        if m == 0:
            _C = np.sqrt(n + 1)
        else:
            _C = np.sqrt(2 * n + 1)
        for k in range(kmax + 1):
            _R += ((-1) ** k * factorial(n - k) /
                   (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                    factorial(0.5 * (n - abs(m)) - k)) *
                   rho ** (n - 2 * k))
        if m >= 0:
            _O = np.cos(m * phi)
        if m < 0:
            _O = - np.sin(m * phi)
        return _C * _R * _O

    @staticmethod
    def _zernike_derivatives(n, m, rho, phi):
        if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
            raise ValueError("n and m are not valid Zernike indices")
        kmax = int((n - abs(m)) / 2)
        _R = 0
        _dR = 0
        _O = 0
        _dO = 0
        for k in range(kmax + 1):
            _R += ((-1) ** k * factorial(n - k) /
                   (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                    factorial(0.5 * (n - abs(m)) - k)) *
                   rho ** (n - 2 * k))
            _dR += ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial(0.5 * (n + abs(m)) - k) *
                     factorial(0.5 * (n - abs(m)) - k)) *
                    rho ** (n - 2 * k - 1)) * (n - 2 * k)
        if m >= 0:
            _O = np.cos(m * phi)
            _dO = - m * np.sin(m * phi)
        if m < 0:
            _O = - np.sin(m * phi)
            _dO = - m * np.cos(m * phi)
        zdx = _dR * _O * np.cos(phi) - (_R / rho) * _dO * np.sin(phi)
        zdy = _dR * _O * np.sin(phi) + (_R / rho) * _dO * np.cos(phi)
        return zdx, zdy

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


class ReconstructionThread(threading.Thread):
    def __init__(self, recon, callback):
        threading.Thread.__init__(self)
        self.recon = recon
        self.lock = threading.Lock()
        self.is_finished = threading.Event()
        self.callback = callback

    def run(self):
        with self.lock:
            self.recon()
        self.is_finished.set()
        if self.callback is not None:
            self.callback()
