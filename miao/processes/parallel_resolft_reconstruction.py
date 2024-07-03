class ImageReconstruction:

    def __init__(self):
        self.na = 1.4
        self.wl = 0.5
        self.n, self.nx, self.ny = 27 * 27, 1024, 1024
        self.pixel_size = 0.063
        self.dfx, self.dfy = 1 / (self.nx * self.pixel_size), 1 / (self.ny * self.pixel_size)
        self.rbx, self.rby = (self.na / self.wl) / self.dfx, (self.na / self.wl) / self.dfy
        # Parameters
        self.fwhm = self.wl / (2 * self.na)
        self.sigma = fwhm_to_sigma(self.fwhm, self.pixel_size)
        self.period_x_um = 0.83  # um
        self.period_y_um = 0.83  # um
        self.size_x = 1024  # Size of the array in x direction (pixels)
        self.size_y = 1024  # Size of the array in y direction (pixels)
        self.range_x_um = (20.26, 43.504)  # Range in x direction in micrometers
        self.range_y_um = (20.26, 43.504)  # Range in y direction in micrometers
        self.threshold = 0.5  # Threshold value

        # Convert periods and ranges to pixel coordinates
        self.period_x = int(self.period_x_um / self.pixel_size)
        self.period_y = int(self.period_y_um / self.pixel_size)
        self.range_x = (self.range_x_um[0] / self.pixel_size, self.range_x_um[1] / self.pixel_size)
        self.range_y = (self.range_y_um[0] / self.pixel_size, self.range_y_um[1] / self.pixel_size)

        # Create the 2D array
        self.array = np.zeros((self.size_y, self.size_x))

        # Create grid of points, include endpoint
        self.x_coords = np.arange(self.range_x[0], self.range_x[1] + 1, self.period_x)
        self.y_coords = np.arange(self.range_y[0], self.range_y[1] + 1, self.period_y)

        # Create a Gaussian kernel
        self.kernel_size = 6 * int(self.sigma) + 1

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
                x_end = min(self.size_x, x_idx + self.kernel_size // 2 + 1)
                y_end = min(self.size_y, y_idx + self.kernel_size // 2 + 1)
                self.array[y_start:y_end, x_start:x_end] += gaussian_k[
                                                       (self.kernel_size // 2 - (y_idx - y_start)):(
                                                               self.kernel_size // 2 + (y_end - y_idx)),
                                                       (self.kernel_size // 2 - (x_idx - x_start)):(
                                                               self.kernel_size // 2 + (x_end - x_idx))
                                                       ]

        # Apply the threshold
        self.array[self.array < self.threshold] = 0

        # Crop sub-arrays and stack into a 3D array
        stacked_subarrays = []
        center_positions = []

        for x_center in self.x_coords:
            for y_center in self.y_coords:
                x_idx = int(x_center)
                y_idx = int(y_center)
                x_start = max(0, x_idx - self.period_x // 2)
                y_start = max(0, y_idx - self.period_y // 2)
                x_end = min(self.size_x, x_start + self.period_x)
                y_end = min(self.size_y, y_start + self.period_y)
                subarray = self.array[y_start:y_end, x_start:x_end]

                # Ensure the subarray is of the desired size
                if subarray.shape == (self.period_y, self.period_x):
                    stacked_subarrays.append(subarray)
                    center_positions.append((x_center * self.pixel_size, y_center * self.pixel_size))

        stacked_subarrays = np.array(stacked_subarrays)
        center_positions = np.array(center_positions)


def gaussian_kernel(size, sigma, center):
    """Returns a 2D Gaussian kernel with a given center."""
    x = np.linspace(-size // 2, size // 2, size) - center[0]
    y = np.linspace(-size // 2, size // 2, size) - center[1]
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel


def fwhm_to_sigma(fwhm, pixel_size):
    return (fwhm / pixel_size) / (2 * np.sqrt(2 * np.log(2)))


if __name__ == "__main__":
    import tifffile as tf
    import numpy as np
    from scipy.ndimage import gaussian_filter, convolve
    from scipy.interpolate import RectBivariateSpline

    gain_value = 2.5
    window_radius = int(1 * np.ceil((0.5 / 2.8) / 0.0785))

    I_in = tf.imread(r"C:\Users\ruizhe.lin\Desktop\20240530161327_dot_scanning_crop-1.tif")

    # Upscale factor calculation
    number_row_initial, number_column_initial = I_in.shape
    scale_factor = 2.5
    number_row_scaleup = int(scale_factor * number_row_initial)
    number_column_scaleup = int(scale_factor * number_column_initial)

    x0 = np.linspace(-0.5, 0.5, number_column_initial)
    y0 = np.linspace(-0.5, 0.5, number_row_initial)
    X0, Y0 = np.meshgrid(x0, y0)
    x = np.linspace(-0.5, 0.5, number_column_scaleup)
    y = np.linspace(-0.5, 0.5, number_row_scaleup)
    X, Y = np.meshgrid(x, y)

    # DPR on single frames
    single_frame_I_in = I_in - np.min(I_in)
    local_minimum = np.zeros_like(single_frame_I_in)
    single_frame_I_in_localmin = np.zeros_like(single_frame_I_in)

    for u in range(number_row_initial):
        for v in range(number_column_initial):
            sub_window = single_frame_I_in[
                         max(0, u - window_radius):min(number_row_initial, u + window_radius + 1),
                         max(0, v - window_radius):min(number_column_initial, v + window_radius + 1)
                         ]
            local_minimum[u, v] = np.min(sub_window)
            single_frame_I_in_localmin[u, v] = single_frame_I_in[u, v] - local_minimum[u, v]

    # Upscale using spline interpolation
    interp_localmin = RectBivariateSpline(y0, x0, single_frame_I_in_localmin)
    single_frame_localmin_magnified = interp_localmin(y, x, grid=True)
    single_frame_localmin_magnified[single_frame_localmin_magnified < 0] = 0
    single_frame_localmin_magnified = np.pad(single_frame_localmin_magnified, ((10, 10), (10, 10)), mode='constant')

    interp_in = RectBivariateSpline(y0, x0, single_frame_I_in)
    single_frame_I_magnified = interp_in(y, x, grid=True)
    single_frame_I_magnified[single_frame_I_magnified < 0] = 0
    single_frame_I_magnified = np.pad(single_frame_I_magnified, ((10, 10), (10, 10)), mode='constant')

    number_row, number_column = single_frame_I_magnified.shape

    # Local normalization
    I_normalized = single_frame_localmin_magnified / (gaussian_filter(single_frame_localmin_magnified, 8) + 1e-5)

    # Calculate normalized gradients
    sobelX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_y = convolve(I_normalized, sobelX, mode='reflect')
    gradient_x = convolve(I_normalized, sobelY, mode='reflect')

    gradient_x = gradient_x / (I_normalized + 1e-5)
    gradient_y = gradient_y / (I_normalized + 1e-5)

    # Calculate pixel displacements
    displacement_x = gain_value * gradient_x
    displacement_y = gain_value * gradient_y
    displacement_x[np.abs(displacement_x) > 10] = 0
    displacement_y[np.abs(displacement_y) > 10] = 0

    # Calculate I_out with weighted pixel displacements
    single_frame_I_out = np.zeros((number_row, number_column))
    for nx in range(10, number_row - 10):
        for ny in range(10, number_column - 10):
            weighted1 = (1 - np.abs(displacement_x[nx, ny] - np.fix(displacement_x[nx, ny]))) * (
                    1 - np.abs(displacement_y[nx, ny] - np.fix(displacement_y[nx, ny])))
            weighted2 = (1 - np.abs(displacement_x[nx, ny] - np.fix(displacement_x[nx, ny]))) * (
                np.abs(displacement_y[nx, ny] - np.fix(displacement_y[nx, ny])))
            weighted3 = (np.abs(displacement_x[nx, ny] - np.fix(displacement_x[nx, ny]))) * (
                    1 - np.abs(displacement_y[nx, ny] - np.fix(displacement_y[nx, ny])))
            weighted4 = (np.abs(displacement_x[nx, ny] - np.fix(displacement_x[nx, ny]))) * (
                np.abs(displacement_y[nx, ny] - np.fix(displacement_y[nx, ny])))

            coordinate1 = [int(np.fix(displacement_x[nx, ny])), int(np.fix(displacement_y[nx, ny]))]
            coordinate2 = [int(np.fix(displacement_x[nx, ny])),
                           int(np.fix(displacement_y[nx, ny]) + np.sign(displacement_y[nx, ny]))]
            coordinate3 = [int(np.fix(displacement_x[nx, ny]) + np.sign(displacement_x[nx, ny])),
                           int(np.fix(displacement_y[nx, ny]))]
            coordinate4 = [int(np.fix(displacement_x[nx, ny]) + np.sign(displacement_x[nx, ny])),
                           int(np.fix(displacement_y[nx, ny]) + np.sign(displacement_y[nx, ny]))]

            single_frame_I_out[nx + coordinate1[0], ny + coordinate1[1]] += weighted1 * single_frame_I_magnified[
                nx, ny]
            single_frame_I_out[nx + coordinate2[0], ny + coordinate2[1]] += weighted2 * single_frame_I_magnified[
                nx, ny]
            single_frame_I_out[nx + coordinate3[0], ny + coordinate3[1]] += weighted3 * single_frame_I_magnified[
                nx, ny]
            single_frame_I_out[nx + coordinate4[0], ny + coordinate4[1]] += weighted4 * single_frame_I_magnified[
                nx, ny]

    single_frame_I_out = single_frame_I_out[10:-10, 10:-10]
    single_frame_I_magnified = single_frame_I_magnified[10:-10, 10:-10]
    result = [single_frame_I_out, single_frame_I_magnified]
    tf.imwrite(r"C:\Users\ruizhe.lin\Desktop\result.tif", np.asarray(result))
