import numpy as np


# Function to perform the Weighted Gerchberg-Saxton Algorithm
def weighted_gerchberg_saxton(mag_image, mag_fourier, weight_image=1.0, weight_fourier=1.0, iterations=100):
    # Random initial phase in spatial domain
    phase_image = np.random.rand(*mag_image.shape) * 2 * np.pi

    # Complex field in the spatial domain
    image_complex = mag_image * np.exp(1j * phase_image)

    for i in range(iterations):
        # Forward Fourier transform
        ft = np.fft.fft2(image_complex)

        # Enforce known Fourier magnitude with weighting
        phase_fourier = np.angle(ft)
        ft = (mag_fourier * np.exp(1j * phase_fourier)) * weight_fourier + ft * (1 - weight_fourier)

        # Inverse Fourier transform
        image_complex = np.fft.ifft2(ft)

        # Enforce known image magnitude with weighting
        phase_image = np.angle(image_complex)
        image_complex = (mag_image * np.exp(1j * phase_image)) * weight_image + image_complex * (1 - weight_image)

    return image_complex


def quantified_metric(image):
    uni = 1 - (image.max() - image.min()) / (image.max() + image.min())
    sig = 100 * np.sqrt(np.average((image - np.average(image)) ** 2)) / np.average(image)
    return image.sum(), uni, sig


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def generate_focal_spot_array(shape, num_foci_x, num_foci_y):
        # Create an empty Fourier domain pattern
        target_pattern = np.zeros(shape, dtype=np.complex64)

        # Define spacing between the focal spots
        spacing_x = shape[1] // (num_foci_x + 1)
        spacing_y = shape[0] // (num_foci_y + 1)

        # Set focal spots in the Fourier domain
        for i in range(1, num_foci_x + 1):
            for j in range(1, num_foci_y + 1):
                target_pattern[j * spacing_y, i * spacing_x] = 1.0  # Set intensity for each focal spot

        return target_pattern


    shape = (512, 512)  # Size of the phase mask
    num_foci_x = 10
    num_foci_y = 10
    target_pattern = generate_focal_spot_array(shape, num_foci_x, num_foci_y)

    fourier_image = np.fft.fft2(target_pattern)
    mag_image = np.abs(target_pattern)  # Known magnitude in spatial domain
    mag_fourier = np.abs(fourier_image)  # Known magnitude in Fourier domain

    weight_image = 0.8  # Weighting factor for the spatial domain
    weight_fourier = 0.9  # Weighting factor for the Fourier domain

    reconstructed_image = weighted_gerchberg_saxton(mag_image, mag_fourier, weight_image, weight_fourier,
                                                    iterations=100)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(np.abs(target_pattern), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Reconstructed Phase')
    plt.imshow(np.angle(reconstructed_image), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Reconstructed Magnitude')
    plt.imshow(np.abs(reconstructed_image), cmap='gray')

    plt.show()
