import numpy as np
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.filters import maximum_filter
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


class BeadScanReconstruction:

    def __init__(self):
        self.dx = 13 / (63 * 3)  # micron
        self.na = 1.4
        self.wl = 0.505
        self.d = int(np.ceil((self.wl / (2 * self.na)) / self.dx))
        self.neighborhood_size = 16
        self.threshold = 2000

    def find_beads(self, neighborhood_size, threshold, verbose=False):
        data = self.imgstack[0]
        data_max = maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) / 2
            y_center = (dy.start + dy.stop - 1) / 2
            if x_center > (self.d + self.r) and y_center > (self.d + self.r):
                x.append(int(x_center))
                y.append(int(y_center))
        if verbose:
            plt.imshow(data)
            plt.autoscale(False)
            plt.plot(x, y, 'ro')
        return x, y

    def single_bead_recontruction(self, x, y):
        img = self.imgstack[:, int(y - (self.d + self.r)):int(y + self.d), int(x - (self.d + self.r)):int(x + self.d)]
        result = np.zeros((self.ns + 1, self.ns + 1))
        for j in range(self.ns + 1):
            for i in range(self.ns + 1):
                temp = img[(self.ns + 1) * i + j]
                indy, indx = np.where((temp == temp.max()))
                signal = temp[indy[0] - 1:indy[0] + 2, indx[0] - 1:indx[0] + 2].sum()
                result[j, i] = signal
        return result

    def reconstruct_all_beads(self, data, stepsize):
        self.imgstack = data
        self.n, self.nx, self.ny = self.imgstack.shape
        self.ns = int(np.sqrt(self.n) - 1)
        self.ss = stepsize
        self.r = int(np.ceil(self.ns * self.ss / self.dx))
        x, y = self.find_beads(self.neighborhood_size, self.threshold)
        self.result = np.zeros((len(x), self.ns + 1, self.ns + 1))
        for l in range(len(x)):
            self.result[l] = self.single_bead_recontruction(x[l], y[l])
        self.final_image = np.zeros((self.nx, self.ny))
        s = int(np.floor((self.ns + 1) / 2))
        for l in range(len(x)):
            self.final_image[y[l] - s:y[l] + s + 1, x[l] - s:x[l] + s + 1] = self.result[l]
