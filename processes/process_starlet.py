import numpy as np
import tifffile as tf


class StarletTransform:

    def __init__(self):
        self.c = []

    def input_image(self, file):
        img = tf.imread(file)
        self.img_shape = img.shape
        self.c.append(img)

    def cubic_kernel(self):
        self._kernel = np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
        self._kernel_size = self._kernel.shape[0]

    def discrete_convolution(self, j):
        temp = np.zeros(self.img_shape)
        for m in range(self._kernel_size):
            for n in range(self._kernel_size):
                temp += self._kernel[m] * self._kernel[n] * np.roll(np.roll(self.c[j], shift=n, axis=0), shift=m,
                                                                    axis=1)
        temp1 = np.zeros(self.img_shape)
        for m in range(5):
            for n in range(5):
                temp1 = temp1 + self._kernel[m] * self._kernel[n] * np.roll(np.roll(temp, shift=n, axis=0), shift=m,
                                                                            axis=1)
        self.c.append(self.c[j] - temp1)
