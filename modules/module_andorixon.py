import sys
import threading
from collections import deque

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
                self.imageSize = self.xpixels * self.ypixels
                print("Detector size: xpixels = {} ypixels = {}".format(self.xpixels, self.ypixels))
            else:
                print(atmcd_errors.Error_Codes(self.ret))
            self.data = FixedLengthList(16)
            self.camera_thread = None
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
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            gain = self.get_gain()
            print("Set CCD EMGain to {}".format(gain))
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_gain(self):
        (self.ret, gain) = self.sdk.GetEMCCDGain()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            return gain
        else:
            print(atmcd_errors.Error_Codes(self.ret))
            return 0

    def set_roi(self, hbin, vbin, hstart, hend, vstart, vend):
        self.ret = self.sdk.SetImage(hbin, vbin, hstart, hend, vstart, vend)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("hbin = {} \nvbin = {} \nhstart = {} \nhend = {} \nvstart = {} \nvend = {}".format(
                hbin, vbin, hstart, hend, vstart, vend))
            self.xpixels = vend - vstart + 1
            self.ypixels = hend - hstart + 1
            self.imageSize = self.xpixels * self.ypixels
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
        (self.ret, self.fminExposure, self.fAccumulate, self.fKinetic) = self.sdk.GetAcquisitionTimings()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Get Acquisition Timings exposure = {} accumulate = {} kinetic = {}".format(self.fminExposure,
                                                                                              self.fAccumulate,
                                                                                              self.fKinetic))
        else:
            print(atmcd_errors.Error_Codes(self.ret))
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
        self.camera_thread = CameraThread(self)

    def start_live(self):
        self.ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.data.data_list.clear()
            self.camera_thread.start()
            print('Start live image')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def get_last_image(self):
        if not self.data.is_empty():
            return self.data.get_last_element()

    def get_images(self):
        self.imageSize = self.xpixels * self.ypixels
        (self.ret, self.first, self.last) = self.sdk.GetNumberNewImages()
        num = self.last - self.first + 1
        (self.ret, self.arr, self.valid_first, self.valid_last) = self.sdk.GetImages16(self.first, self.last,
                                                                                       self.imageSize * num)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            self.arr = self.arr.reshape(num, self.imageSize)
            for n in range(num):
                self.data.add_element(self.arr[n].reshape(self.xpixels, self.ypixels))

    def stop_live(self):
        self.camera_thread.stop()
        self.camera_thread = None
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
        (self.ret, self.minExposure, self.Accumulate, self.Kinetic) = self.sdk.GetAcquisitionTimings()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Retrieve Acquisition Timings: exposure = {} accumulate = {} kinetic = {}'.format(self.minExposure,
                                                                                                    self.Accumulate,
                                                                                                    self.Kinetic))
        else:
            print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.SetNumberKinetics(num)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print("Set Number of Kinetics to {}".format(num))
        else:
            print(atmcd_errors.Error_Codes(self.ret))
        self.ret = self.sdk.PrepareAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Ready to acquire data')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def start_data_acquisition(self):
        self.ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print('Kinetic acquisition start')
        else:
            print(atmcd_errors.Error_Codes(self.ret))

    def check_acquisition_progress(self):
        (self.ret, self.numoAccumulate, self.numoKinetics) = self.sdk.GetAcquisitionProgress()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
            print(
                "GetAcquisitionProgress returned {} \n"
                "number of accumulations completed = {} \n"
                "kinetic scans completed = {}".format(self.ret, self.numoAccumulate, self.numoKinetics))

    # def get_images(self, num):
    #     self.imageSize = self.xpixels * self.ypixels
    #     (self.ret, self.arr) = self.sdk.GetAcquiredData16(num * self.imageSize)
    #     if atmcd_errors.Error_Codes.DRV_SUCCESS == self.ret:
    #         self.data = self.arr.reshape(num, self.xpixels, self.ypixels)
    #         print('Data self.retrieved')
    #     else:
    #         print(atmcd_errors.Error_Codes(self.ret))

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


class CameraThread(threading.Thread):
    def __init__(self, cam):
        threading.Thread.__init__(self)
        self.cam = cam
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                self.cam.get_images()

    def stop(self):
        self.running = False
        self.join()


class FixedLengthList:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data_list = deque(maxlen=max_length)

    def add_element(self, element):
        if len(self.data_list) == self.max_length:
            self.data_list.clear()
        self.data_list.append(element)

    def get_elements(self):
        return list(self.data_list)

    def get_last_element(self):
        return self.data_list[-1] if self.data_list else None

    def is_empty(self):
        return len(self.data_list) == 0
