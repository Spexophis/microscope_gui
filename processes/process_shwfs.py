import os
import numpy as np
import tifffile as tf
from scipy.signal import fftconvolve as corr
from skimage.filters import threshold_otsu
from tools import tool_improc as ipr
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
pi = np.pi


class WavefrontSensing:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.n_lenslets_x = 18
        self.n_lenslets_y = 19
        self.n_lenslets = self.n_lenslets_x * self.n_lenslets_y
        self.x_center_base = 1316
        self.y_center_base = 1370
        self.x_center_offset = 1316
        self.y_center_offset = 1369
        self.lenslet_spacing = 40  # spacing between each lenslet
        self.hsp = 16  # size of subimage is 2 * hsp
        self.calfactor = (.0065 / 5.2) * 150  # pixel size * focalLength * pitch
        self.method = 'correlation'
        self.mag = 1
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        sectioncorr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(sectioncorr.argmax(), sectioncorr.shape)
        self._ref = None
        self._meas = None
        self.wf = np.array([])

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @property
    def ref(self):
        return self._ref

    @ref.setter
    def ref(self, new_ref):
        self.logg.info(f"Changing shwfs base")
        self._ref = new_ref

    @property
    def meas(self):
        return self._meas

    @meas.setter
    def meas(self, new_meas):
        # self.logg.info(f"Changing shwfs offset")
        self._meas = new_meas

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

    def wavefront_reconstruction(self, md='correlation', rt=False):
        (gradx, grady) = self.get_gradient_xy(mtd=md)
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

    def get_gradient_xy(self, mtd='correlation'):
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
        base = self._sub_back(self.ref, 0.8)
        offset = self._sub_back(self.meas, 0.8)
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
                if mtd == 'correlation':
                    seccorr = corr(1.0 * secbase, 1.0 * sec[::-1, ::-1], mode='full')
                    py, px = self._parabolic_fit(seccorr)
                    gradx[iy, ix] = (self.CorrCenter[1] - px) * self.calfactor
                    grady[iy, ix] = (self.CorrCenter[0] - py) * self.calfactor
                elif mtd == 'centerofmass':
                    sy, sx = self._center_of_mass(secbase)
                    py, px = self._center_of_mass(sec)
                    gradx[iy, ix] = (px - sx) * self.calfactor
                    grady[iy, ix] = (py - sy) * self.calfactor
        return (gradx, grady)

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
    def _sub_back(img, factor):
        thresh = factor * threshold_otsu(img)
        binary = img > thresh
        return (img - thresh) * binary

    @staticmethod
    def _elliptical_mask(radius, size):
        coord_x = np.arange(0.5, size[0], 1.0)
        coord_y = np.arange(0.5, size[1], 1.0)
        y, x = np.meshgrid(coord_y, coord_x)
        x -= size[0] / 2.
        y -= size[1] / 2.
        return (x * x / (radius[0] * radius[0])) + (y * y / (radius[1] * radius[1])) <= 1

    def generate_influence_matrix(self, data_folder, dm_info, method='phase', sv=False, verbose=False):
        n_actuators, amp, n_zernikes, zslopes = dm_info
        if method == 'phase':
            _influence_matrix = np.zeros((self.n_lenslets, n_actuators))
            wfs = np.zeros((n_actuators, self.n_lenslets_y, self.n_lenslets_x))
        elif method == 'zonal':
            _influence_matrix = np.zeros((2 * self.n_lenslets, n_actuators))
        elif method == 'modal':
            _influence_matrix = np.zeros((n_zernikes, n_actuators))
        else:
            raise ValueError("Invalid method")
        _msk = self._elliptical_mask((self.n_lenslets_y / 2, self.n_lenslets_x / 2),
                                     (self.n_lenslets_y, self.n_lenslets_x))
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif") & filename.startswith("actuator"):
                ind = int(filename.split("_")[1])
                if verbose:
                    print(filename.split("_")[1])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                n, x, y = data_stack.shape
                if n != 4:
                    raise "The image number has to be 4"
                if method == 'phase':
                    self.ref, self.meas = data_stack[0], data_stack[1]
                    wfp = self.wavefront_reconstruction(rt=True)
                    wfs[ind] = wfp
                    # wfn = self.wavefront_reconstruction(data_stack[2], data_stack[3], rt=True)
                    # _influence_matrix[:, ind] = ((wfp - wfn) / (2 * amp)).reshape(self.n_lenslets)
                    msk = (wfp != 0.0).astype(np.float32)
                    mn = wfp.sum() / msk.sum()
                    wfp = msk * (wfp - mn)
                    _influence_matrix[:, ind] = wfp.reshape(self.n_lenslets)
                else:
                    self.ref, self.meas = data_stack[0], data_stack[1]
                    gdxp, gdyp = self.get_gradient_xy()
                    self.ref, self.meas = data_stack[2], data_stack[3]
                    gdxn, gdyn = self.get_gradient_xy()
                    if method == 'zonal':
                        _influence_matrix[:self.n_lenslets, ind] = ((gdxp - gdxn) / (2 * amp)).reshape(self.n_lenslets)
                        _influence_matrix[self.n_lenslets:, ind] = ((gdyp - gdyn) / (2 * amp)).reshape(self.n_lenslets)
                    if method == 'modal':
                        a1 = ipr.get_eigen_coefficients(np.concatenate((gdxp.flatten(), gdyp.flatten())), zslopes)
                        a2 = ipr.get_eigen_coefficients(np.concatenate((gdxn.flatten(), gdyn.flatten())), zslopes)
                        _influence_matrix[:, ind] = ((a1 - a2) / (2 * amp)).flatten()
        _control_matrix = ipr.pseudo_inverse(_influence_matrix, n=32)
        if sv:
            tf.imwrite(os.path.join(data_folder, f"influence_function_{method}.tif"), _influence_matrix)
            tf.imwrite(os.path.join(data_folder, f"control_matrix_{method}.tif"), _control_matrix)
            if 'wfs' in locals():
                if isinstance(wfs, np.ndarray):
                    tf.imwrite(os.path.join(data_folder, "influence_function_images.tif"), wfs)
