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
