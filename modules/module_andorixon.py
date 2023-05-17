import sys

from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors

sys.path.append(r'C:\Program Files\Andor SDK')


class EMCCDCamera:

    def __init__(self):

        self.sdk = atmcd(r'C:\Program Files\Andor SDK')  # Load the atmcd library
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
            self.ret = self.sdk.SetReadMode(atmcd_codes.Read_Mode.IMAGE)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print("Set Read Mode to Image")
            else:
                print(atmcd_errors.Error_Codes(self.ret))
            self.ret = self.sdk.SetTriggerMode(atmcd_codes.Trigger_Mode.EXTERNAL_EXPOSURE_BULB)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print("Set TriggerMode to External Exposure")
            else:
                print(atmcd_errors.Error_Codes(self.ret))
            (self.ret, self.xpixels, self.ypixels) = self.sdk.GetDetector()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                print("Detector size: xpixels = {} ypixels = {}".format(self.xpixels, self.ypixels))
            else:
                print(atmcd_errors.Error_Codes(self.ret))

            self.xpixels = None
            self.ypixels = None
            self.imageSize = None
            self.data = None
            self.ps = 13  # micron
        else:
            print('AndorEMCCD is not initiated')

    def close(self):
        self.cooler_off()
        self.ret = self.sdk.ShutDown()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Andor EMCCD Shut Down")
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def cooler_on(self):
        (self.ret, self.cooler_status) = self.sdk.IsCoolerOn()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            if not self.cooler_status:
                self.ret = self.sdk.CoolerON()
                if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                    print("Cooler ON")
                else:
                    print(atmcd_errors.Error_Codes(self.ret))
            else:
                print("Cooler is ON")
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def cooler_off(self):
        (self.ret, self.cooler_status) = self.sdk.IsCoolerOn()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            if self.cooler_status:
                self.ret = self.sdk.CoolerOFF()
                if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                    print("Cooler OFF")
                else:
                    print(atmcd_errors.Error_Codes(self.ret))
            else:
                print("Cooler is OFF")
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def check_camera_status(self):
        (self.ret, self.status) = self.sdk.GetStatus()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print(atmcd_errors.Error_Codes(self.status))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_ccd_temperature(self):
        (self.ret, temperature) = self.sdk.GetTemperature()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            return temperature
        else:
            print(atmcd_errors.Error_Codes(self.ret))
            return temperature

    def set_gain(self, emccdgain):
        self.ret = self.sdk.SetEMCCDGain(emccdgain)
        if atmcd_errors.Error_Codes.DRV_SUCCESS != self.ret:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_gain(self):
        try:
            (self.ret, gain) = self.sdk.GetEMCCDGain()
            if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
                return gain
            else:
                print(atmcd_errors.Error_Codes(self.ret))
        except Exception as e:
            print("An error occurred:", e)

    def set_roi(self, hbin, vbin, hstart, hend, vstart, vend):
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
        self.ret = self.sdk.SetAcquisitionMode(atmcd_codes.Acquisition_Mode.RUN_TILL_ABORT)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Set Acquisition Mode to Run Till Abort")
        else:
            print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.SetKineticCycleTime(0)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Set Kinetic Cycle Time to 0")
        else:
            print(atmcd_errors.Error_Codes(self.ret))
        # self.ret = self.sdk.SetImage(1, 1, 1, self.xpixels, 1, self.ypixels)
        # print("Function SetImage returned {} hbin = 1 vbin = 1 hstart = 1 hend = {} vstart = 1 vend = {}".format(
        #     self.ret, self.xpixels, self.ypixels))

    def start_live(self):
        self.ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Start live image')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_last_image(self):
        self.imageSize = self.xpixels * self.ypixels
        (self.ret, self.arr) = self.sdk.GetMostRecentImage16(self.imageSize)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.data = self.arr.reshape(self.xpixels, self.ypixels)
            return True
        else:
            print(atmcd_errors.Error_Codes(self.ret))
            return False

    def stop_live(self):
        self.ret = self.sdk.AbortAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Live image stopped')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def prepare_data_acquisition(self, num):
        self.ret = self.sdk.SetAcquisitionMode(atmcd_codes.Acquisition_Mode.KINETICS)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Successfully set Acquisition Mode to KINETICS')
        else:
            print(atmcd_errors.Error_Codes(self.ret))
        (self.ret, self.xpixels, self.ypixels) = self.sdk.GetDetector()
        print("Function GetDetector returned {} xpixels = {} ypixels = {}".format(self.ret, self.xpixels, self.ypixels))
        self.ret = self.sdk.SetImage(1, 1, 1, self.xpixels, 1, self.ypixels)
        print("Function SetImage returned {} hbin = 1 vbin = 1 hstart = 1 hend = {} vstart = 1 vend = {}".format(
            self.ret, self.xpixels, self.ypixels))
        (self.ret, self.fminExposure, self.fAccumulate, self.fKinetic) = self.sdk.GetAcquisitionTimings()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Successfully retrieved Acquisition Timings')
        else:
            print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.SetNumberKinetics(num)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Ready to acquire data')
        else:
            print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.PrepareAcquisition()

    def start_data_acquisition(self):
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
            "GetAcquisitionProgress returned {} \n"
            "number of accumulations completed = {} \n"
            "kinetic scans completed = {}".format(self.ret, self.numofaccumulate, self.numofkinetics))

    def get_images(self, num):
        self.imageSize = self.xpixels * self.ypixels
        (self.ret, self.arr) = self.sdk.GetAcquiredData16(num * self.imageSize)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.data = self.arr.reshape(num, self.xpixels, self.ypixels)
            print('Data self.retrieved')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def finish_data_acquisition(self):
        self.ret = self.sdk.AbortAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Single acquisition done')
            self.free_memory()
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def free_memory(self):
        self.ret = self.sdk.FreeInternalMemory()
        print("FreeInternalMemory returned {}".format(self.ret))
        print(atmcd_errors.Error_Codes(self.ret))
