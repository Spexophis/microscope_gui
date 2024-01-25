import numpy as np

wl = 0.5  # wavelength in microns
na = 1.4  # numerical aperture
dx = 13 / (2 * 5 * 63 / 3)  # pixel size in microns
nx = 1024  # size of region
fs = 1 / dx  # Spatial sampling frequency, inverse microns
df = fs / nx  # Spacing between discrete frequency coordinates, inverse microns
radius = (na / wl) / df


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
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(data))))


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
    g = pv * np.fromfunction(lambda i, j: np.exp(-((i - ux) ** 2. + (j - uy) ** 2.) / (2. * sigma ** 2.)), (_nx, _ny))
    return g


def snr(img, lpr, hpr, gau=False):
    _ny, _nx = img.shape
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
