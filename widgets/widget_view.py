import numpy as np
from PyQt5 import QtWidgets
import naparitools


class ViewWidget(QtWidgets.QWidget):
    """ Widget containing viewbox that displays the new detector frames. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        naparitools.addNapariGrayclipColormap()
        self.napariViewer = naparitools.EmbeddedNapari()

        self.imgLayers = {}

        self.viewCtrlLayout = QtWidgets.QVBoxLayout()
        self.viewCtrlLayout.addWidget(self.napariViewer.get_widget())
        self.setLayout(self.viewCtrlLayout)
        
        self.name_wf = 'Wavefront'
        self.imgLayers[self.name_wf] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_wf, blending='additive',
            colormap=None, protected=True)
        
        self.name_sh = 'ShackHartmann'
        self.imgLayers[self.name_sh] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_sh, blending='additive',
            colormap=None, protected=True)
        
        self.name_fft = 'FFT'
        self.imgLayers[self.name_fft] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_fft, blending='additive',
            colormap=None, protected=True)
        
        self.name_m = 'Main Camera'
        self.imgLayers[self.name_m] = self.napariViewer.add_image(
            np.zeros((1024, 1024)), rgb=False, name=self.name_m, blending='additive',
            colormap=None, protected=True)    
        

    def getImage(self, name):
        return self.imgLayers[name].data

    def setImage(self, name, im):
        self.imgLayers[name].data = im

    def clearImage(self, name):
        self.setImage(name, np.zeros((512, 512)))

    def resetView(self):
        self.napariViewer.reset_view()


# Copyright (C) 2020-2021 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.