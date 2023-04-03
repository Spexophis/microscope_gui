# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:17:06 2022

@author: ruizhe.lin
"""

class AOController():

    def __init__(self, view):
        self.view = view
    
    
    def getexposuretime(self):
        return self.view.QDoubleSpinBox_exposuretime.value()
    
    def getparameters(self):
        return self.view.QSpinBox_base_xcenter.value(), self.view.QSpinBox_base_ycenter.value(),\
            self.view.QSpinBox_offset_xcenter.value(), self.view.QSpinBox_offset_ycenter.value(),\
            self.view.QSpinBox_diamter.value(), self.view.QSpinBox_radius.value(), self.view.QSpinBox_spacing.value()
        
    def getacturator(self):
        return self.view.QSpinBox_actuator.value(), self.view.QDoubleSpinBox_actuator_push.value()
    
    def getzernikemode(self):
        return self.view.QSpinBox_zernike_mode.value(), self.view.QDoubleSpinBox_zernike_mode_amp.value()

    def get_file_name(self):
        return self.view.QLineEdit_filename.text()
    
    def get_ao_iteration(self):
        return self.view.QSpinBox_zernike_mode_start.value(), self.view.QSpinBox_zernike_mode_stop.value(),\
            self.view.QDoubleSpinBox_zernike_mode_amps_start.value(), self.view.QDoubleSpinBox_zernike_mode_amps_step.value(), self.view.QSpinBox_zernike_mode_amps_stepnum.value(),\
    
    def get_ao_parameters(self):
        return self.view.QDoubleSpinBox_lpf.value(), self.view.QDoubleSpinBox_hpf.value(), self.view.QComboBox_metric.currentIndex(), self.view.QComboBox_metric.currentText()
        