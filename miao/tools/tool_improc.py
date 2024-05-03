import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import cv2
from skimage import filters
import matplotlib.pyplot as plt

wl = 0.5  # wavelength in microns
na = 1.4  # numerical aperture
dx = 0.0785  # pixel size in microns
fs = 1 / dx  # Spatial sampling frequency, inverse microns


def rms(data):
    _nx, _ny = data.shape
    _n = _nx * _ny
    m = np.mean(data, dtype=np.float64)
    a = (data - m) ** 2
    r = np.sqrt(np.sum(a) / _n)
    return r


def img_properties(img):
    return img.min(), img.max(), rms(img)


def fourier_transform(data):
    return np.log(np.abs(fftshift(fft2(data))))


def disc_array(shape=(128, 128), radi=64.0, origin=None, dtp=np.float64):
    _nx = shape[0]
    _ny = shape[1]
    ox = _nx / 2
    oy = _ny / 2
    x = np.linspace(-ox, ox - 1, _nx)
    y = np.linspace(-oy, oy - 1, _ny)
    xv, yv = np.meshgrid(x, y)
    rho = np.sqrt(xv ** 2 + yv ** 2)
    disc = (rho < radi).astype(dtp)
    if origin is not None:
        s0 = origin[0] - int(_nx / 2)
        s1 = origin[1] - int(_ny / 2)
        disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
    return disc


def gaussian_filter(shape, sigma, pv, orig=None):
    _nx, _ny = shape
    if orig is None:
        ux = _nx / 2.
        uy = _ny / 2.
    else:
        ux, uy = orig
    g = np.fromfunction(lambda i, j: np.exp(-((i - ux) ** 2. + (j - uy) ** 2.) / (2. * sigma ** 2.)), (_nx, _ny))
    return pv * g


def snr(img, lpr, hpr, gau=False):
    _ny, _nx = img.shape
    df = fs / _nx
    radius = (na / wl) / df
    m = img.min()
    img[img <= m] = 0.
    img[img > m] = img[img > m] - m
    if gau:
        lp = gaussian_filter(shape=(_nx, _ny), sigma=lpr * radius, pv=1, orig=None)
        hp = 1 - gaussian_filter(shape=(_nx, _ny), sigma=hpr * radius, pv=1, orig=None)
    else:
        lp = disc_array(shape=(_nx, _ny), radi=lpr * radius)
        hp = disc_array(shape=(_nx, _ny), radi=0.9 * radius) - disc_array(shape=(_nx, _ny), radi=hpr * radius)
    aft = np.fft.fftshift(np.fft.fft2(img))
    return (np.abs(hp * aft)).sum() / (np.abs(lp * aft)).sum()


def hpf(img, hpr, gau=True):
    _nx, _ny = img.shape
    df = fs / _nx
    radius = (na / wl) / df
    m = img.min()
    img[img <= m] = 0.
    img[img > m] = img[img > m] - m
    if gau:
        hp = 1 - gaussian_filter(shape=(_nx, _ny), sigma=hpr * radius, pv=1, orig=None)
    else:
        hp = disc_array(shape=(_nx, _ny), radi=0.9 * radius) - disc_array(shape=(_nx, _ny), radi=hpr * radius)
    aft = np.fft.fftshift(np.fft.fft2(img))
    aft = aft * hp
    return (np.abs(aft)).sum()


def peak_find(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    a, b, c = np.polyfit(x, y, 2)
    p = -1 * b / a / 2.0
    if a > 0:
        raise ValueError("no maximum")
    elif (p >= x.max()) or (p <= x.min()):
        raise ValueError("maximum exceeding range")
    else:
        return p


def get_profile(data, ax):
    data = data - data.min()
    data = data / data.max()
    if ax == 'X':
        return data.mean(0)
    elif ax == 'Y':
        return data.mean(1)
    else:
        raise ValueError("invalid axis")


def pseudo_inverse(A, n=32):
    u, s, vt = np.linalg.svd(A)
    s_inv = np.zeros_like(A.T)
    if n is None:
        s_inv[:min(A.shape), :min(A.shape)] = np.diag(1 / s[:min(A.shape)])
    else:
        s_inv[:n, :n] = np.diag(1 / s[:n])
    return vt.T @ s_inv @ u.T


def get_eigen_coefficients(mta, mtb, ng=32):
    mp = pseudo_inverse(mtb, n=ng)
    return np.matmul(mp, mta)


def calculate_focus_measure(image):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var


def calculate_focus_measure_with_sobel(image):
    edges = filters.sobel(image)
    focus_measure = np.var(edges)
    return focus_measure


def calculate_focus_measure_with_laplacian(image):
    laplacian_image = filters.laplace(image)
    focus_measure = np.var(laplacian_image)
    return focus_measure


def meshgrid(nx_, ny_):
    x_ = np.arange(-nx_ / 2, nx_ / 2)
    y_ = np.arange(-ny_ / 2, ny_ / 2)
    xv_, yv_ = np.meshgrid(x_, y_, indexing='ij', sparse=True)
    return np.roll(xv_, int(nx_ / 2)), np.roll(yv_, int(ny_ / 2))


def shift(arr, shifts=None):
    if shifts is None:
        shifts = np.array(arr.shape) / 2
    if len(arr.shape) == len(shifts):
        for m, p in enumerate(shifts):
            arr = np.roll(arr, int(p), m)
    return arr


def get_pupil(nx_, rd_):
    msk = shift(disc_array(shape=(nx_, nx_), radi=rd_)) / np.sqrt(np.pi * rd_ ** 2) / nx_
    phi = np.zeros((nx_, nx_))
    return msk * np.exp(1j * phi)


def get_psf(nx_, rd_):
    bpp = get_pupil(nx_, rd_)
    psf_ = np.abs((fft2(fftshift(bpp)))) ** 2
    return psf_ / psf_.sum()


def get_correlation(image_to_be_computed, shift_orientation, shift_spacing, xv, yv, psf):
    kx = dx * np.cos(shift_orientation) / shift_spacing
    ky = dx * np.sin(shift_orientation) / shift_spacing
    ysh = np.exp(2j * np.pi * (kx * xv + ky * yv))
    otf0 = fft2(psf)
    imgf0 = fft2(image_to_be_computed)
    otf = fft2(psf * ysh)
    imgf = fft2(image_to_be_computed * ysh)
    a = np.sum(imgf0 * np.conj(otf0) * np.conj(imgf * np.conj(otf)))
    return np.abs(a), np.angle(a)


def find_pattern(data, angle=0., spacing=0.268, nps=10, r_ang=0.005, r_sp=0.005, verbose=False):
    nx, ny = data.shape
    df = fs / nx
    radius = (na / wl) / df
    xv_, yv_ = meshgrid(nx, nx)
    psf_ = get_psf(nx, radius)
    d_ang = 2 * r_ang / nps
    d_sp = 2 * r_sp / nps
    ang_iter = np.arange(-r_ang, r_ang + d_ang / 2, d_ang) + angle
    sp_iter = np.arange(-r_sp, r_sp + d_sp / 2, d_sp) + spacing
    mags = np.zeros((nps + 1, nps + 1))
    phs = np.zeros((nps + 1, nps + 1))
    for m, ang in enumerate(ang_iter):
        for n, sp in enumerate(sp_iter):
            mag, phase = get_correlation(data, ang, sp, xv_, yv_, psf_)
            if np.isnan(mag):
                mags[m, n] = 0.0
            else:
                mags[m, n] = mag
                phs[m, n] = phase
    if verbose:
        plt.figure()  # Set the figure size for better visualization
        plt.subplot(211)  # First subplot for magnitudes
        plt.imshow(mags, vmin=mags.min(), vmax=mags.max(),
                   extent=(sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()),
                   interpolation='none', cmap='viridis')
        plt.colorbar()
        plt.title('Magnitude')
        plt.xlabel('Spatial Iteration')
        plt.ylabel('Angular Iteration')
        plt.subplot(212)
        plt.imshow(phs, extent=(sp_iter.min(), sp_iter.max(), ang_iter.max(), ang_iter.min()),
                   interpolation='none', cmap='twilight')
        plt.colorbar()
        plt.title('Phase')
        plt.xlabel('Spatial Iteration')
        plt.ylabel('Angular Iteration')
        plt.tight_layout()
        plt.show()
    m, n = np.where(mags == mags.max())
    ang_max = m[0] * d_ang - r_ang + angle
    sp_max = n[0] * d_sp - r_sp + spacing
    return ang_max, sp_max, mags[m, n]
