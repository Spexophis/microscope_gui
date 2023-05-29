class ViewController:

    def __init__(self, view):
        self.v = view

    def plot_main(self, data):
        self.v.show_image('Main Camera', data)

    def plot_fft(self, data):
        self.v.show_image('FFT', data)

    def plot_sh(self, data):
        self.v.show_image('ShackHartmann', data)

    def plot_wf(self, data):
        self.v.show_image('Wavefront', data)

    def get_image_data(self, layer):
        return self.v.get_image(layer)

    def plot(self, data):
        self.v.plot(data)

    def plot_update(self, data):
        self.v.update_plot(data)
