import numpy as np
import tifffile as tf
from numpy.fft import fft2, fftshift
from skimage.feature import peak_local_max


class ImageReconstruction:

    def __init__(self):
        self.na = 1.4
        self.wl = 0.5
        self.resolution = self.wl / (2 * self.na)
        self.pixel_size = 0.063
        self.sigma = self.resolution / (2 * np.sqrt(2 * np.log(2)))

    def load_data(self, fd):
        self.data_stack = tf.imread(fd)
        self.n, self.ny, self.nx = self.data_stack.shape
        self.xv, self.yv = np.meshgrid(np.linspace(0, self.nx - 1, self.nx), np.linspace(0, self.ny - 1, self.ny))
        self.xv = self.pixel_size * self.xv
        self.yv = self.pixel_size * self.yv

    def set_scanning_parameters(self, step_nums=(32, 32)):
        assert step_nums[0] * step_nums[1] == self.n, f"Scanning step numbers does not match the data size"
        self.step_y, self.step_x = step_nums

    def set_focal_parameters(self, periods=(0.83, 0.83), ranges=((20., 40.), (20., 40.))):
        self.period_x_um, self.period_y_um = periods  # micrometer
        self.range_x_um, self.range_y_um = ranges  # micrometers
        self.x_centers = np.arange(self.range_x_um[0], self.range_x_um[1], self.period_x_um)
        self.y_centers = np.arange(self.range_y_um[0], self.range_y_um[1], self.period_y_um)

    def generate_center_array(self):
        center_array = np.zeros_like(self.xv)
        for xc in self.x_centers:
            for yc in self.y_centers:
                x_idx = (np.abs(self.xv[0, :] - xc)).argmin()
                y_idx = (np.abs(self.yv[:, 0] - yc)).argmin()
                center_array[y_idx, x_idx] = 1
        return center_array

    def create_gaussian_1d_array(self, x_=True):
        array = np.zeros((self.ny, self.nx))
        if x_:
            for x_center in self.x_centers:
                array += self.gaussian_1d(self.xv, x_center, self.sigma)
        else:
            for y_center in self.y_centers:
                array += self.gaussian_1d(self.yv, y_center, self.sigma)
        return array

    def create_gaussian_2d_array(self):
        array = np.zeros((self.ny, self.nx))
        for x_center in self.x_centers:
            for y_center in self.y_centers:
                array += self.gaussian_2d(self.xv, self.yv, x_center, y_center, self.sigma)
        return array

    def apply_gaussian(self, stack):
        mask = self.create_gaussian_2d_array()
        assert stack.shape[1:] == mask.shape, "Gaussian mask shape must match the shape of each 2D slice in the stack"
        masked_stack = stack * mask[np.newaxis, :, :]
        return masked_stack

    def stack_subarray(self, array_stack):
        subarray_stack = []
        for y_center in self.y_centers:
            for x_center in self.x_centers:
                x_start = max(0, x_center - self.period_x_um / 2)
                y_start = max(0, y_center - self.period_y_um / 2)
                x_end = min(self.nx * self.pixel_size, x_start + self.period_x_um)
                y_end = min(self.ny * self.pixel_size, y_start + self.period_y_um)
                x_start = int(x_start / self.pixel_size)
                y_start = int(y_start / self.pixel_size)
                x_end = int(x_end / self.pixel_size)
                y_end = int(y_end / self.pixel_size)
                subarray = array_stack[:, y_start:y_end, x_start:x_end]
                subarray = np.sum(subarray, axis=(1, 2)).reshape(self.step_y, self.step_x)
                subarray_stack.append(subarray)
        return np.asarray(subarray_stack)

    def get_result(self, substack):
        assert substack.shape == (
            self.x_centers.shape[0] * self.y_centers.shape[0], self.step_y,
            self.step_x), f"Input stack has the wrong shape"
        reshaped_stack = substack.reshape(self.y_centers.shape[0], self.x_centers.shape[0], self.step_y, self.step_x)
        transposed_stack = reshaped_stack.transpose(0, 2, 1, 3)
        tiled_array = transposed_stack.reshape(self.y_centers.shape[0] * self.step_y,
                                               self.x_centers.shape[0] * self.step_x)
        return tiled_array

    def extract_periods(self):
        image = np.average(self.data_stack, axis=0)
        fft_image = fftshift(fft2(image))
        magnitude_spectrum = np.abs(fft_image)
        normalized_spectrum = np.log1p(magnitude_spectrum)
        # Identify peaks in the magnitude spectrum
        peaks = peak_local_max(normalized_spectrum, min_distance=10, threshold_rel=0.1)
        # Compute the distances from the center of the Fourier image
        center = np.array(normalized_spectrum.shape) // 2
        peak_distances = np.sqrt((peaks[:, 0] - center[0]) ** 2 + (peaks[:, 1] - center[1]) ** 2)
        # Sort distances and select the most prominent peaks
        sorted_indices = np.argsort(peak_distances)
        sorted_peaks = peaks[sorted_indices]
        # Compute the periods corresponding to the prominent peaks
        periods = []
        for peak in sorted_peaks[1:5]:  # Consider the two most prominent peaks
            distance = np.sqrt((peak[0] - center[0]) ** 2 + (peak[1] - center[1]) ** 2)
            period = (image.shape[0] / distance) * self.pixel_size
            periods.append(period)
        return periods, normalized_spectrum, sorted_peaks[1:5]

    @staticmethod
    def gaussian_1d(x_, mu_x, sigma):
        g = np.exp(-((x_ - mu_x) ** 2) / (2 * sigma ** 2))
        msk = ((x_ - mu_x) ** 2) / (2 * sigma ** 2)
        msk = msk <= 1.
        return g * msk

    @staticmethod
    def gaussian_2d(x_, y_, mu_x, mu_y, sigma):
        g = np.exp(-((x_ - mu_x) ** 2 + (y_ - mu_y) ** 2) / (2 * sigma ** 2))
        msk = ((x_ - mu_x) ** 2 + (y_ - mu_y) ** 2) / (2 * sigma ** 2)
        msk = msk <= 1.
        return g * msk


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r = ImageReconstruction()
    r.pixel_size = 0.0785
    r.load_data(r"C:\Users\ruizhe.lin\Desktop\20240605223219_dot_scanning_crop.tif")
    r.set_focal_parameters(periods=(0.821, 0.8), ranges=((0.12, r.nx * r.pixel_size), (0.5, r.ny * r.pixel_size)))
    array = r.generate_center_array()
    data_avg = np.average(r.data_stack, axis=0)
    plt.figure()
    plt.imshow(array, cmap='viridis', interpolation='none')
    plt.imshow(data_avg, cmap='plasma', interpolation='none', alpha=0.4)
    plt.show(block=False)
