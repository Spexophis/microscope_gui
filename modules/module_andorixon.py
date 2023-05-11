import sys

import numpy as np
from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors

sys.path.append(r'C:\Program Files\Andor SDK')


class EMCCDCamera:

    def __init__(self):

        self.sdk = atmcd(r'C:\Program Files\Andor SDK')  # Load the atmcd library
        self.codes = atmcd_codes
        ret = self.sdk.Initialize(r'C:/Program Files/Andor SDK/atmcd64d.dll')  # Initialize camera      
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:

            (ret, self.iSerialNumber) = self.sdk.GetCameraSerialNumber()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                print("Camera Serial Number : {}".format(self.iSerialNumber))
            else:
                print(atmcd_errors.Error_Codes(ret))

            ret = self.sdk.CoolerON()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                print("Cooler ON")
                ret = self.sdk.SetTemperature(-60)
                if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                    print("Set target temperature to -60")
                else:
                    print(atmcd_errors.Error_Codes(ret))
            else:
                print(atmcd_errors.Error_Codes(ret))

            ret = self.sdk.SetReadMode(self.codes.Read_Mode.IMAGE)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                print("Set Read Mode to Image")
            else:
                print(atmcd_errors.Error_Codes(ret))

            ret = self.sdk.SetTriggerMode(self.codes.Trigger_Mode.EXTERNAL_EXPOSURE_BULB)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                print("Set TriggerMode to External Exposure")
            else:
                print(atmcd_errors.Error_Codes(ret))

            (ret, self.xpixels, self.ypixels) = self.sdk.GetDetector()
            self.image_size = self.xpixels * self.ypixels
            if atmcd_errors.Error_Codes.DRV_SUCCESS != ret:
                print(atmcd_errors.Error_Codes(ret))

            self.data = None
            self.ps = 13  # micron
        else:
            print('AndorEMCCD is not initiated')

    def close(self):
        ret = self.sdk.ShutDown()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Andor EMCCD Shut Down")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def check_camera_status(self):
        (ret, self.status) = self.sdk.GetStatus()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print(atmcd_errors.Error_Codes(self.status))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_ccd_temperature(self):
        (ret, self.temperature) = self.sdk.GetTemperature()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            return self.temperature
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_gain(self, emccdgain):
        ret = self.sdk.SetEMCCDGain(emccdgain)
        if atmcd_errors.Error_Codes.DRV_SUCCESS != ret:
            print(atmcd_errors.Error_Codes(ret))

    def get_gain(self):
        try:
            (ret, gain) = self.sdk.GetEMCCDGain()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                return gain
            else:
                print(atmcd_errors.Error_Codes(ret))
        except Exception as e:
            print("An error occurred:", e)

    def set_roi(self, hbin, vbin, hstart, hend, vstart, vend):
        ret = self.sdk.SetImage(hbin, vbin, hstart, hend, vstart, vend)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("hbin = {} \nvbin = {} \nhstart = {} \nhend = {} \nvstart = {} \nvend = {}".format(
                hbin, vbin, hstart, hend, vstart, vend))
            self.xpixels = vend - vstart + 1
            self.ypixels = hend - hstart + 1
            self.ps = 13 / hbin
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_trigger_mode(self, ind):
        ret = self.sdk.SetTriggerMode(ind)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Trigger Mode Set")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_exposure_time(self, exposure):
        ret = self.sdk.SetExposureTime(exposure)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Exposure Time to {}".format(exposure))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def prepare_live(self):
        ret = self.sdk.SetAcquisitionMode(self.codes.Acquisition_Mode.RUN_TILL_ABORT)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            ret = self.sdk.PrepareAcquisition()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                print('Ready to live image')
            else:
                print(atmcd_errors.Error_Codes(ret))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def start_live(self):
        ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Start live image')
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_last_image(self):
        try:
            (ret, self.arr) = self.sdk.GetMostRecentImage16(self.image_size)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                self.data = self.arr.reshape(self.xpixels, self.ypixels)
                return True
            else:
                print(atmcd_errors.Error_Codes(ret))
                return False
        except Exception as e:
            print("An error occurred:", e)

    def stop_live(self):
        ret = self.sdk.AbortAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Live image stopped')
        else:
            print(atmcd_errors.Error_Codes(ret))

    def prepare_data_acquisition(self, num):
        ret = self.sdk.SetAcquisitionMode(self.codes.Acquisition_Mode.KINETICS)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Successfully set Acquisition Mode to KINETICS')
            (ret, self.fminExposure, self.fAccumulate, self.fKinetic) = self.sdk.GetAcquisitionTimings()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                print('Successfully retrieved Acquisition Timings')
                ret = self.sdk.SetNumberKinetics(num)
                if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                    print('Ready to acquire data')
                else:
                    print(atmcd_errors.Error_Codes(ret))
            else:
                print(atmcd_errors.Error_Codes(ret))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def start_data_acquisition(self):
        ret = self.sdk.PrepareAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            ret = self.sdk.StartAcquisition()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                print('Kinetic acquisition start')
            else:
                print(atmcd_errors.Error_Codes(ret))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def check_acquisition_progress(self):
        (ret, self.numofaccumulate, self.numofkinetics) = self.sdk.GetAcquisitionProgress()
        print(
            "GetAcquisitionProgress returned {} \nnumber of accumulations completed = {} \nkinetic scans completed = {}".format(
                ret, self.numofaccumulate, self.numofkinetics))

    def get_images(self, num):
        (ret, self.arr) = self.sdk.GetAcquiredData16(num * self.image_size)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            self.data = self.arr.reshape(num, self.xpixels, self.ypixels)
            print('Data retrieved')
        else:
            print(atmcd_errors.Error_Codes(ret))

    def finish_data_acquisition(self):
        ret = self.sdk.AbortAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Single acquisition done')
            self.free_memory()
        else:
            print(atmcd_errors.Error_Codes(ret))

    def free_memory(self):
        ret = self.sdk.FreeInternalMemory()
        print("FreeInternalMemory returned {}".format(ret))
        print(atmcd_errors.Error_Codes(ret))
