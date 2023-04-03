# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:20:36 2022

@author: ruizhe.lin
"""

import pycobolt


class CoboltLaser():
    
    def __init__(self):
        super().__init__()
        
        try:
            self.l405 = pycobolt.CoboltLaser(port='COM13')
            print('405 nm Laser Connected')
            self.l405_handle = True
        except:
            print("405 nm Laser Not Found")
            self.l405_handle = False
        try:
            self.l488_0 = pycobolt.CoboltLaser(port='COM12')
            print('488 nm Laser Connected')
            self.l488_0_handle = True
        except:
            print("488 nm Laser Not Found")
            self.l488_0_handle = False
        try:
            self.l488_1 = pycobolt.CoboltLaser(port='COM14')
            print('488 nm Laser #1 Connected')
            self.l488_1_handle = True
        except:
            print("488 nm Laser #1 Not Found")
            self.l488_1_handle = False
        try:
            self.l488_2 = pycobolt.CoboltLaser(port='COM4')
            print('488 nm Laser #2 Connected')
            self.l488_2_handle = True
        except:
            print("488 nm Laser #2 Not Found")
            self.l488_2_handle = False
            
        
    def __del__(self):
        if self.l405_handle:
            del self.l405
        if self.l488_0_handle:
            del self.l488_0
        if self.l488_1_handle:
            del self.l488_1
        if self.l488_2_handle:
            del self.l488_2

    def all_off(self):
        if self.l405_handle:
            self.l405.send_cmd('l0')    
        if self.l488_0_handle:
            self.l488_0.send_cmd('l0')
        if self.l488_1_handle:
            self.l488_1.send_cmd('l0')
        if self.l488_2_handle:
            self.l488_2.send_cmd('l0')
        
    def all_on(self):
        if self.l405_handle:
            self.l405.send_cmd('l1')
        if self.l488_0_handle:
            self.l488_0.send_cmd('l1')
        if self.l488_1_handle:
            self.l488_1.send_cmd('l1')
        if self.l488_2_handle:
            self.l488_2.send_cmd('l1')
    
    def laserON_488_0(self):
        if self.l488_0_handle:
            self.l488_0.send_cmd('l1')
    
    def laserON_488_1(self):
        if self.l488_1_handle:
            self.l488_1.send_cmd('l1')

    def laserON_488_2(self):
        if self.l488_2_handle:
            self.l488_2.send_cmd('l1')
                        
    def laserON_405(self):
        if self.l405_handle:
            self.l405.send_cmd('l1')

    def laserOFF_488_0(self):
        if self.l488_0_handle:
            self.l488_0.send_cmd('l0')

    def laserOFF_488_1(self):
        if self.l488_1_handle:
            self.l488_1.send_cmd('l0')
            
    def laserOFF_488_2(self):
        if self.l488_2_handle:
            self.l488_2.send_cmd('l0')
            
    def laserOFF_405(self):
        if self.l405_handle:
            self.l405.send_cmd('l0')
            
    def constant_power_488_0(self, p488_0):
        if self.l488_0_handle:
            self.l488_0.constant_power(p488_0)
    
    def constant_power_488_1(self, p488_1):
        if self.l488_1_handle:
            self.l488_1.constant_power(p488_1)
            
    def constant_power_488_2(self, p488_2):
        if self.l488_2_handle:
            self.l488_2.constant_power(p488_2)
                       
    def constant_power_405(self, p405):
        if self.l405_handle:
            self.l405.constant_power(p405)
            
    def modulation_mode_488_0(self, p488_0):
        if self.l488_0_handle:
            self.l488_0.modulation_mode(p488_0)
            self.l488_0.digital_modulation(enable=1)
            
    def modulation_mode_488_1(self, p488_1):
        if self.l488_1_handle:
            self.l488_1.modulation_mode(p488_1)
            self.l488_1.digital_modulation(enable=1)
            
    def modulation_mode_488_2(self, p488_2):
        if self.l488_2_handle:
            self.l488_2.modulation_mode(p488_2)
            self.l488_2.digital_modulation(enable=1)
            
    def modulation_mode_405(self, p405):
        if self.l405_handle:
            self.l405.modulation_mode(p405)
            self.l405.digital_modulation(enable=1)          
