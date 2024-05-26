class ViewController:

    def __init__(self, view):
        self.v = view

    def plot_main(self, data, layer=0):
        self.v.show_image(self.v.img_layers[layer], data)

    def plot_sh(self, data, layer=1):
        self.v.show_image(self.v.img_layers[layer], data)

    def plot_fft(self, data, layer="FFT"):
        self.v.show_image(self.v.img_layers[4], data)

    def plot_shb(self, data, layer="ShackHartmann(Base)"):
        self.v.show_image(self.v.img_layers[5], data)

    def plot_wf(self, data, layer="Wavefront"):
        self.v.show_image(self.v.img_layers[6], data)

    def get_image_data(self, layer=1):
        return self.v.get_image(self.v.img_layers[layer])

    def plot(self, data, x=None, s=None):
        self.v.plot(data, x, s)

    def plot_update(self, data, x=None, s=None):
        self.v.update_plot(data, x, s)

    def display_metrics(self, m1, m2, m3):
        self.v.QLCDNumber_img_laplacian_cv2.display(m1)
        self.v.QLCDNumber_img_laplacian_scikit.display(m2)
        self.v.QLCDNumber_img_sobel_scikit.display(m3)
