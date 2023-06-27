import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Qt5Agg')


class Smat:

    def __init__(self, stack, radius=None):
        self.wfstack = stack.copy()
        nz, nx, ny = self.wfstack.shape
        print(nz)
        if not (nz == 97):
            raise 'There must be 97 images!'
        # set size of array
        if radius is None:
            radius = 64
        self.sz = 2 * radius
        # cut out wavefront
        #        R0 = self.wfstack[:,(nx/2-radius):(nx/2+radius),(nx/2-radius):(nx/2+radius)]
        R0 = self.wfstack
        msk = (R0[0] != 0.0).astype(np.float32)
        nz, nx, ny = R0.shape
        # remove mean
        for m in range(nz):
            mn = R0[m].sum() / msk.sum()
            R0[m] = msk * (R0[m] - mn)
        R = np.zeros((nx * ny, nz))
        for m in range(nz):
            R[:, m] = R0[m].reshape(nx * ny)
        self.R = R
        self.calcS(64)

    def calcS(self, Ns=21):  # SVD decomposition of R. Use largest Ns vectors.
        """ Ns is number of singular values to retain """
        u, s, vh = np.linalg.svd(self.R, 0, 1)
        ut = np.transpose(u)
        self.s = s
        sd = 1. / s
        s = np.zeros((97, 97))
        for i in range(Ns):
            s[i, i] = sd[i]
        v = np.transpose(vh)
        t = np.dot(v, s)
        self.u = u
        self.v = v
        self.S = np.dot(t, ut)

    def calcSalpha(self, Ns=21, alpha=2.0):  # SVD decomposition of R. Use largest Ns vectors.
        """ Ns is number of singular values to retain """
        u, s, vh = np.linalg.svd(self.R, 0, 1)
        ut = np.transpose(u)
        self.s = s
        sd = 1. / np.sqrt(s ** 2 + alpha ** 2)
        s = np.zeros((69, 69))
        for i in range(Ns):
            s[i, i] = sd[i]
        v = np.transpose(vh)
        t = np.dot(v, s)
        self.u = u
        self.v = v
        self.S = np.dot(t, ut)

    def view_eigen_vectors(self):
        t = (self.u.reshape(self.sz, self.sz, 69)).swapaxes(1, 2).swapaxes(0, 1)
        plt.imshow(t, vmin=None)
        plt.plot(self.s)
