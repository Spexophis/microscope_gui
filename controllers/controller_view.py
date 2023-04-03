class ViewController():

    def __init__(self, view):
        self.view = view

    def plot_main(self, data):
        self.view.setImage('Main Camera', data)
        
    def plot_fft(self, data):
        self.view.setImage('FFT', data)
    
    def plot_sh(self, data):
        self.view.setImage('ShackHartmann', data)
    
    def plot_wf(self, data):
        self.view.setImage('Wavefront', data)
