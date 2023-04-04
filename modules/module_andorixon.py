from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors
import numpy as np


class EMCCDCamera:

    def __init__(self):

        self.sdk = atmcd(r'C:\Program Files\Andor SDK')  # Load the atmcd library
        self.codes = atmcd_codes
        self.ret = self.sdk.Initialize(r'C:/Program Files/Andor SDK/atmcd64d.dll')  # Initialize camera      
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:

            (self.ret, self.iSerialNumber) = self.sdk.GetCameraSerialNumber()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print("Camera Serial Number : {}".format(self.iSerialNumber))
            else:
                print(atmcd_errors.Error_Codes(self.ret))

            self.ret = self.sdk.CoolerON()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print("Cooler ON")
                self.ret = self.sdk.SetTemperature(-60)
                if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                    print("Set target temperature to -60")
                else:
                    print(atmcd_errors.Error_Codes(self.ret))
            else:
                print(atmcd_errors.Error_Codes(self.ret))

            self.ret = self.sdk.SetReadMode(self.codes.Read_Mode.IMAGE)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print("Set Read Mode to Image")
            else:
                print(atmcd_errors.Error_Codes(self.ret))

            self.ret = self.sdk.SetTriggerMode(self.codes.Trigger_Mode.EXTERNAL_EXPOSURE_BULB)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print("Set TriggerMode to External Exposure")
            else:
                print(atmcd_errors.Error_Codes(self.ret))

            (self.ret, self.xpixels, self.ypixels) = self.sdk.GetDetector()
            if atmcd_errors.Error_Codes.DRV_SUCCESS != self.ret:
                print(atmcd_errors.Error_Codes(self.ret))

            self.ps = 13  # micron

    def shutdown(self):
        self.ret = self.sdk.ShutDown()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Andor EMCCD Shut Down")
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def check_camera_status(self):
        (self.ret, self.status) = self.sdk.GetStatus()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print(atmcd_errors.Error_Codes(self.status))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_ccd_temperature(self):
        (self.ret, self.temperature) = self.sdk.GetTemperature()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            return self.temperature
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def set_emccd_gain(self, emccdgain):
        self.ret = self.sdk.SetEMCCDGain(emccdgain)
        if atmcd_errors.Error_Codes.DRV_SUCCESS != self.ret:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_emccd_gain(self):
        (self.ret, self.emgain) = self.sdk.GetEMCCDGain()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            return self.emgain
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    # def set_readout_mode(self):
    #     self.ret = self.sdk.SetReadMode(self.codes.Read_Mode.FULL_VERTICAL_BINNING)
    #     SetReadMode(3)
    #     SetSingleTrack(128,20)

    #     SetReadMode(1)
    #     SetMultiTrack(5,20,0,bottom, gap)

    def set_image(self, hbin, vbin, hstart, hend, vstart, vend):
        self.ret = self.sdk.SetImage(hbin, vbin, hstart, hend, vstart, vend)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("hbin = {} \nvbin = {} \nhstart = {} \nhend = {} \nvstart = {} \nvend = {}".format(
                hbin, vbin, hstart, hend, vstart, vend))
            self.xpixels = vend - vstart + 1
            self.ypixels = hend - hstart + 1
            self.ps = 13 / hbin
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def set_trigger_mode(self, ind):
        self.ret = self.sdk.SetTriggerMode(ind)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Trigger Mode Set")
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def set_exposure_time(self, exposure):
        self.ret = self.sdk.SetExposureTime(exposure)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Set Exposure Time to {}".format(exposure))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def prepare_live(self):
        self.ret = self.sdk.SetAcquisitionMode(self.codes.Acquisition_Mode.RUN_TILL_ABORT)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.ret = self.sdk.PrepareAcquisition()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print('Ready to live image')
            else:
                print(atmcd_errors.Error_Codes(self.ret))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def start_live(self):
        self.ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Start live image')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def getImage_live(self):
        self.imageSize = self.xpixels * self.ypixels
        (self.ret, self.arr) = self.sdk.GetMostRecentImage16(self.imageSize)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.data = self.arr.reshape(self.xpixels, self.ypixels)
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def stop_live(self):
        self.ret = self.sdk.AbortAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Live image stopped')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def prepare_single_acquisition(self):
        self.ret = self.sdk.SetAcquisitionMode(self.codes.Acquisition_Mode.SINGLE_SCAN)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.ret = self.sdk.PrepareAcquisition()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print('Ready for single acquisition')
            else:
                print(atmcd_errors.Error_Codes(self.ret))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def single_acquisition(self):
        self.ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Start single acquisition')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_acquired_image(self):
        self.ret = self.sdk.WaitForAcquisition()
        self.imageSize = self.xpixels * self.ypixels
        (self.ret, self.arr, self.validfirst, self.validlast) = self.sdk.GetImages16(1, 1, self.imageSize)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.data = self.arr.reshape(self.xpixels, self.ypixels)
            print('Single acquisition done')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def stop_acquisition(self):
        self.ret = self.sdk.AbortAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Single acquisition done')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def prepare_kinetic_acquisition(self, num):
        self.ret = self.sdk.SetAcquisitionMode(self.codes.Acquisition_Mode.KINETICS)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Successfully set Acquisition Mode to KINETICS')
            (self.ret, self.fminExposure, self.fAccumulate, self.fKinetic) = self.sdk.GetAcquisitionTimings()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print('Successfully retrieved Acquisition Timings')
                self.ret = self.sdk.SetNumberKinetics(num)
                if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                    print('Ready to acquire data')
                else:
                    print(atmcd_errors.Error_Codes(self.ret))
            else:
                print(atmcd_errors.Error_Codes(self.ret))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def start_kinetic_acquisition(self):
        self.ret = self.sdk.PrepareAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.ret = self.sdk.StartAcquisition()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print('Kinetic acquisition start')
            else:
                print(atmcd_errors.Error_Codes(self.ret))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def check_acquisition_progress(self):
        (self.ret, self.numofaccumulate, self.numofkinetics) = self.sdk.GetAcquisitionProgress()
        print(
            "GetAcquisitionProgress returned {} \nnumber of accumulations completed = {} \nkinetic scans completed = {}".format(
                self.ret, self.numofaccumulate, self.numofkinetics))

    def get_data(self, num):
        self.pixelnum = num * self.xpixels * self.ypixels
        (self.ret, self.arr) = self.sdk.GetAcquiredData16(self.pixelnum)
        self.data = self.arr.reshape(num, self.xpixels, self.ypixels)

    def free_memory(self):
        self.ret = self.sdk.FreeInternalMemory()
        print("FreeInternalMemoryreturned {}".format(self.ret))
        print(atmcd_errors.Error_Codes(self.ret))

    def set_spool(self):
        (self.ret, self.buffer) = self.sdk.GetSizeOfCircularBuffer()
        print("GetSizeOfCircularBuffer returned {} \nBuffer Image Number = {}".format(
            self.ret, self.buffer))
        self.ret = self.sdk.SetSpool(1, 4, r'C:\Users\Testa4\Documents\PythonProjects\ao_resolft\data', self.buffer)
        self.ret = self.sdk.SetSpoolThreadCount(4)

    def get_available_images(self):
        (self.ret, self.cfirst, self.clast) = self.sdk.GetNumberNewImages()
        imageSize = (self.clast - self.cfirst + 1) * self.xpixels * self.ypixels
        (self.ret, self.arr, self.validfirst, self.validlast) = self.sdk.GetImages16(self.cfirst, self.clast, imageSize)
        return self.cfirst, self.clast, self.arr.reshape(self.clast - self.cfirst + 1, self.xpixels, self.ypixels)

    def prepare_sequential_acquisition(self, exposure):
        self.ret = self.sdk.SetAcquisitionMode(self.codes.Acquisition_Mode.SINGLE_SCAN)
        print("SetAcquisitionMode returned {} mode = Single Scan".format(self.ret))
        print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.SetReadMode(self.codes.Read_Mode.IMAGE)
        print("SetReadMode returned {} mode = Image".format(self.ret))
        print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.SetTriggerMode(self.codes.Trigger_Mode.EXTERNAL)
        print("SetTriggerMode returned {} mode = Internal".format(self.ret))
        print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.SetExposureTime(exposure)
        print("SetExposureTime returned {} time = {}".format(self.ret, exposure))
        print(atmcd_errors.Error_Codes(self.ret))
        (self.ret, self.fminExposure, self.fAccumulate, self.fKinetic) = self.sdk.GetAcquisitionTimings()
        print("GetAcquisitionTimings returned {} exposure = {} accumulate = {} kinetic = {}".format(
            self.ret, self.fminExposure, self.fAccumulate, self.fKinetic))

    def sequential_acquisition(self, fnd, num):
        (self.ret, self.buffer) = self.sdk.GetSizeOfCircularBuffer()
        self.sdk.SetSpool(1, 4, fnd, self.buffer)
        self.sdk.SetSpoolThreadCount(4)
        print('Start sequential acquisition')
        result = self.sdk.acquire_series(num)
        self.data = np.asarray(result).reshape(num, self.xpixels, self.ypixels)
        print('Sequential acquisition finished')

    def sequence_acquisition(self, num):
        self.data = []
        for acq in self.sdk.acquire_series(num):
            self.data.append(acq.reshape(self.xpixels, self.ypixels))
