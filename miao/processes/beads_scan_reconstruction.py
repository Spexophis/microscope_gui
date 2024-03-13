import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, label, find_objects


class BeadScanReconstruction:

    def __init__(self):
        self.dx = 13 / (63 * 2.5)  # micron
        self.na = 1.4
        self.wl = 0.505
        self.d = int(np.ceil((self.wl / (2 * self.na)) / self.dx))

    def reconstruct_all_beads(self, data, step_size, threshold, neighborhood_size=32):
        nz, nx, ny = data.shape
        ns = int(np.sqrt(nz) - 1)
        sr = int(np.ceil(ns * step_size / self.dx))
        rd = self.d + sr
        x, y = self.find_beads(data[0], neighborhood_size, threshold, rd)
        final_image = np.zeros((nx, ny))
        s = int((ns + 1) / 2)
        ims = []
        print(len(x))
        for l_ in range(len(x)):
            temp = self.reconstruction_single_bead(data, x[l_], y[l_], ns + 1, rd)
            ims.append(temp)
            if (ns + 1) % 2:
                final_image[y[l_] - s:y[l_] + s + 1, x[l_] - s:x[l_] + s + 1] = temp
            else:
                final_image[y[l_] - s:y[l_] + s, x[l_] - s:x[l_] + s] = temp
        return [x, y], ims, final_image

    @staticmethod
    def find_beads(data, neighborhood_size, threshold, hcr):
        data_max = maximum_filter(data, neighborhood_size)
        data_min = minimum_filter(data, neighborhood_size)
        diff = data_max - data_min
        maxima = (data == data_max) & (diff > threshold)
        labeled, _ = label(maxima)
        slices = find_objects(labeled)
        nx, ny = data.shape
        x, y = [], []
        for slice_ in slices:
            if slice_ is not None:
                slice_center = [(s.start + s.stop - 1) / 2 for s in slice_]
                x_center, y_center = slice_center[1], slice_center[0]
                if hcr < x_center < (nx - hcr) and hcr < y_center < (ny - hcr):
                    x.append(int(x_center))
                    y.append(int(y_center))
        return x, y

    @staticmethod
    def reconstruction_single_bead(imstack, x, y, stpn, hcr):
        img_ = imstack[:, int(y - hcr):int(y + hcr), int(x - hcr):int(x + hcr)]
        result = np.zeros((stpn, stpn))
        for j in range(stpn):
            for i in range(stpn):
                temp = img_[stpn * i + j]
                indy, indx = np.where((temp == temp.max()))
                signal = temp[indy[0] - 1:indy[0] + 2, indx[0] - 1:indx[0] + 2].sum()
                result[j, i] = signal
        return result


if __name__ == '__main__':
    import tifffile as tf
    import pprint
    fn = input("Enter data file directory: ")
    img_stack = tf.imread(fn)
    sz = input("Enter step size: ")
    img_stack = img_stack - img_stack.min()
    hist, bins = np.histogram(img_stack[0], bins=32)
    pprint.pprint(hist)
    pprint.pprint(bins)
    bg = input("Enter background: ")
    img_stack = np.maximum(img_stack - float(bg), 0)
    img = img_stack[0]
    hist, bins = np.histogram(img, bins=16)
    pprint.pprint(hist)
    pprint.pprint(bins)
    thr = input("Enter threshold: ")
    r = BeadScanReconstruction()
    results = r.reconstruct_all_beads(img_stack, float(sz), float(thr), 32)
    fns = input("Enter data file save directory: ")
    if fns == fn:
        fns = input("Enter a different data file save directory: ")
        tf.imwrite(fns, results[2])
    else:
        tf.imwrite(fns, results[2])
