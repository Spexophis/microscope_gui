class ViewController:

    def __init__(self, view):
        self.v = view

    def plot_main(self, data, layer=0):
        self.v.show_image(self.v.img_layers[layer], data)

    def plot_sh(self, data, layer=1):
        self.v.show_image(self.v.img_layers[layer], data)

    def plot_fft(self, data, layer="FFT"):
        self.v.show_image(layer, data)

    def plot_shb(self, data, layer="ShackHartmann(Base)"):
        self.v.show_image(layer, data)

    def plot_wf(self, data, layer="Wavefront"):
        self.v.show_image(layer, data)

    def get_image_data(self, layer=1):
        return self.v.get_image(self.v.img_layers[layer])

    def plot(self, data):
        self.v.plot(data)

    def plot_update(self, data):
        self.v.update_plot(data)
