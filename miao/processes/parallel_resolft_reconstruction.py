import numpy as np
import tifffile as tf


class ImageReconstruction:

    def __init__(self):
        self.na = 1.4
        self.wl = 0.5
        self.resolution = self.wl / (2 * self.na)
        self.pixel_size = 0.063
        # Gaussian kernel
        self.sigma = to_sigma(self.resolution, self.pixel_size)
        self.threshold = 0.5
        self.kernel_size = 6 * int(self.sigma) + 1
        
    def load_data(self, fd):
        data_stack = tf.imread(fd)
        self.n, self.ny, self.nx = data_stack.shape
        self.dfx, self.dfy = 1 / (self.nx * self.pixel_size), 1 / (self.ny * self.pixel_size)
        self.rbx, self.rby = (self.na / self.wl) / self.dfx, (self.na / self.wl) / self.dfy
        
    def set_parameters(self, periods=(0.83, 0.83), ranges=((20.26, 43.504), (20.26, 43.504))):
        self.period_x_um, self.period_y_um = periods  # micrometer
        self.range_x_um, self.range_y_um = ranges  # micrometers
        
        # Convert periods and ranges to pixel coordinates
        self.period_x = int(self.period_x_um / self.pixel_size)
        self.period_y = int(self.period_y_um / self.pixel_size)
        self.range_x = (self.range_x_um[0] / self.pixel_size, self.range_x_um[1] / self.pixel_size)
        self.range_y = (self.range_y_um[0] / self.pixel_size, self.range_y_um[1] / self.pixel_size)

        # Create grid of points
        self.x_coords = np.arange(self.range_x[0], self.range_x[1] + 1, self.period_x)
        self.y_coords = np.arange(self.range_y[0], self.range_y[1] + 1, self.period_y)

    def create_gaussian_array(self):
        array = np.zeros((self.ny, self.nx))
        
        # Place Gaussians on the grid
        for x_center in self.x_coords:
            for y_center in self.y_coords:
                # Calculate sub-pixel offset for Gaussian center
                center = (x_center % 1, y_center % 1)
                gaussian_k = gaussian_kernel(self.kernel_size, self.sigma, center)
                x_idx = int(x_center)
                y_idx = int(y_center)
                # Place the Gaussian in the array, handling edges
                x_start = max(0, x_idx - self.kernel_size // 2)
                y_start = max(0, y_idx - self.kernel_size // 2)
                x_end = min(self.nx, x_idx + self.kernel_size // 2 + 1)
                y_end = min(self.ny, y_idx + self.kernel_size // 2 + 1)
                array[y_start:y_end, x_start:x_end] += gaussian_k[
                                                       (self.kernel_size // 2 - (y_idx - y_start)):(
                                                               self.kernel_size // 2 + (y_end - y_idx)),
                                                       (self.kernel_size // 2 - (x_idx - x_start)):(
                                                               self.kernel_size // 2 + (x_end - x_idx))
                                                       ]

        # Apply the threshold
        array[array < self.threshold] = 0
        return array

    def stack_subarray(self, array):
        # Crop sub-arrays and stack into a 3D array
        stacked_subarrays = []
        center_positions = []

        for x_center in self.x_coords:
            for y_center in self.y_coords:
                x_idx = int(x_center)
                y_idx = int(y_center)
                x_start = max(0, x_idx - self.period_x // 2)
                y_start = max(0, y_idx - self.period_y // 2)
                x_end = min(self.nx, x_start + self.period_x)
                y_end = min(self.ny, y_start + self.period_y)
                subarray = array[y_start:y_end, x_start:x_end]

                # Ensure the subarray is of the desired size
                if subarray.shape == (self.period_y, self.period_x):
                    stacked_subarrays.append(subarray)
                    center_positions.append((x_center * self.pixel_size, y_center * self.pixel_size))

        stacked_subarrays = np.array(stacked_subarrays)
        center_positions = np.array(center_positions)
        return stacked_subarrays, center_positions


def gaussian_kernel(size, sigma, center):
    """Returns a 2D Gaussian kernel with a given center."""
    x = np.linspace(-size // 2, size // 2, size) - center[0]
    y = np.linspace(-size // 2, size // 2, size) - center[1]
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel


def to_sigma(fw, pixel_size):
    return (fw / pixel_size) / (2 * np.sqrt(2 * np.log(2)))


if __name__ == "__main__":
    pass
