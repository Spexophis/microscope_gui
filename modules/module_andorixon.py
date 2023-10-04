import sys
import threading
from collections import deque

import numpy as np
from pyAndorSDK2 import atmcd, atmcd_errors

sys.path.append(r'C:\Program Files\Andor SDK')


class EMCCDCamera:
    temperature = None
    gain = 0
    t_clean = None
    t_readout = None
    t_exposure = None
    t_accumulate = None
    t_kinetic = None
    bin_h = 1
    bin_v = 1
    start_h = 1
    end_h = 1024
    start_v = 1
    end_v = 1024
    pixels_x = 1024
    pixels_y = 1024
    img_size = pixels_x * pixels_y
    ps = 13  # micron
    buffer_size = None
    acq_num = 0
    acq_first = 0
    acq_last = 0
    ind = 0
    data = None
    acq_thread = None

    def __init__(self):

        self.sdk = atmcd(r'C:\Program Files\Andor SDK')  # Load the atmcd library
        ret = self.sdk.Initialize(r'C:/Program Files/Andor SDK/atmcd64d.dll')  # Initialize camera
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            self.get_sn()
            self.cooler_on()
            self.set_frame_transfer(0)
            self.set_readout_rate(0, 3)
        else:
            print('AndorEMCCD is not initiated')

    def __del__(self):
        pass

    def close(self):
        self.cooler_off()
        ret = self.sdk.ShutDown()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Andor EMCCD Shut Down")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_sn(self):
        (ret, iSerialNumber) = self.sdk.GetCameraSerialNumber()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Camera Serial Number : {}".format(iSerialNumber))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def cooler_on(self):
        ret = self.sdk.CoolerON()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Cooler ON")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def cooler_off(self):
        ret = self.sdk.CoolerOFF()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Cooler OFF")
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
            return self.temperature

    def get_sensor_size(self):
        (ret, self.pixels_x, self.pixels_y) = self.sdk.GetDetector()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            self.img_size = self.pixels_x * self.pixels_y
            print("Detector size: pixels_x = {} pixels_y = {}".format(self.pixels_x, self.pixels_y))
        else:
            print(atmcd_errors.Error_Codes(ret))
            return None, None, None

    def set_readout_mode(self, ind):
        """
        0 - Full Vertical Binning
        1 - Multi-Track
        2 - Random-Track
        3 - Single-Track
        4 - Image
        """
        ret = self.sdk.SetReadMode(ind)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Readout Mode")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_readout_rate(self, hs=0, vs=3):
        ret = self.sdk.SetHSSpeed(0, hs)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Horizontal Speed")
        else:
            print(atmcd_errors.Error_Codes(ret))
        ret = self.sdk.SetVSSpeed(vs)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Vertical Speed")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_frame_transfer(self, ft=1):
        ret = self.sdk.SetFrameTransferMode(ft)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Frame Transfer Mode {}".format(ft))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_gain(self):
        ret = self.sdk.SetEMCCDGain(self.gain)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            self.get_gain()
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_gain(self):
        (ret, self.gain) = self.sdk.GetEMCCDGain()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("CCD EMGain is {}".format(self.gain))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_roi(self):
        ret = self.sdk.SetImage(self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("bin_h = {} \nbin_v = {} \nstart_h = {} \nend_h = {} \nstart_v = {} \nend_v = {}".format(
                self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v))
            self.pixels_x = self.end_h - self.start_h + 1
            self.pixels_y = self.end_v - self.start_v + 1
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 13 / self.bin_h
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_trigger_mode(self, ind):
        """
        0 - Internal
        1 - External
        6 - External Start
        7 - External Exposure
        10 - Software
        """
        ret = self.sdk.SetTriggerMode(ind)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Trigger Mode Set")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_exposure_time(self):
        ret = self.sdk.SetExposureTime(self.t_exposure)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Exposure Time to {}".format(self.t_exposure))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_acquisition_mode(self, ind):
        """
        1 - Single Scan
        2 - Accumulate
        3 - Kinetics
        4 - Fast Kinetics
        5 - Run Till Abort
        """
        ret = self.sdk.SetAcquisitionMode(ind)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Acquisition Mode")
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_acquisition_timings(self):
        (ret, self.t_exposure, self.t_accumulate, self.t_kinetic) = self.sdk.GetAcquisitionTimings()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Get Acquisition Timings exposure = {} accumulate = {} kinetic = {}".format(self.t_exposure,
                                                                                              self.t_accumulate,
                                                                                              self.t_kinetic))
        else:
            print(atmcd_errors.Error_Codes(ret))
        (ret, self.t_readout) = self.sdk.GetReadOutTime()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Get Readout Time = {}".format(self.t_readout))
        else:
            print(atmcd_errors.Error_Codes(ret))
        (ret, self.t_clean) = self.sdk.GetKeepCleanTime()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Get Keep Clean Time = {}".format(self.t_clean))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_buffer_size(self):
        (ret, self.buffer_size) = self.sdk.GetSizeOfCircularBuffer()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Get Circular Buffer = {}".format(self.buffer_size))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_kinetic_cycle_time(self, t):
        ret = self.sdk.SetKineticCycleTime(t)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Kinetic Cycle Time to {}".format(t))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def set_kinetics_num(self, kn):
        ret = self.sdk.SetNumberKinetics(kn)
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print("Set Number of Kinetics to {}".format(kn))
        else:
            print(atmcd_errors.Error_Codes(ret))

    def prepare_live(self):
        self.set_readout_mode(4)
        self.set_acquisition_mode(5)
        self.set_trigger_mode(7)
        self.set_kinetic_cycle_time(0)
        self.get_acquisition_timings()
        self.get_buffer_size()
        self.data = DataList(self.buffer_size)
        self.acq_thread = AcquisitionThread(self)

    def start_live(self):
        ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            self.acq_thread.start()
            print('Start live image')
        else:
            print(atmcd_errors.Error_Codes(ret))

    def stop_live(self):
        self.acq_thread.stop()
        self.acq_thread = None
        ret = self.sdk.AbortAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Live image stopped')
            self.data = None
            self.free_memory()
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_images(self):
        (ret, first, last) = self.sdk.GetNumberNewImages()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            num = last - first + 1
            (ret, data_array, valid_first, valid_last) = self.sdk.GetImages16(first, last, self.img_size * num)
            if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
                self.ind = valid_last
                data_array = np.split(data_array.reshape(num, self.img_size), num, axis=0)
                data_array = [subarray.reshape(self.pixels_x, self.pixels_y) for subarray in data_array]
                self.data.add_element(data_array, valid_first, valid_last)
            # else:
            #     print(atmcd_errors.Error_Codes(ret))
        # else:
        #     print(atmcd_errors.Error_Codes(ret))

    def get_last_image(self):
        return self.data.get_last_element()

    def prepare_data_acquisition(self, num):
        self.set_readout_mode(4)
        self.set_acquisition_mode(3)
        self.set_kinetics_num(num)
        self.set_trigger_mode(7)
        # self.set_exposure_time()
        self.set_roi()
        self.get_acquisition_timings()
        self.get_buffer_size()
        ret = self.sdk.PrepareAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Ready to acquire data')
        else:
            print(atmcd_errors.Error_Codes(ret))

    def start_data_acquisition(self):
        ret = self.sdk.StartAcquisition()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Kinetic acquisition start')
        else:
            print(atmcd_errors.Error_Codes(ret))

    def get_acq_num(self):
        (ret, first, last) = self.sdk.GetNumberAvailableImages()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            self.acq_first, self.acq_last = first, last
            print(first, last)
        else:
            print(atmcd_errors.Error_Codes(ret))

    def check_acquisition_progress(self):
        (ret, self.numAccumulate, self.numKinetics) = self.sdk.GetAcquisitionProgress()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print(
                "GetAcquisitionProgress returned {} \n"
                "number of accumulations completed = {} \n"
                "kinetic scans completed = {}".format(ret, self.numAccumulate, self.numKinetics))

    def free_memory(self):
        ret = self.sdk.FreeInternalMemory()
        if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
            print('Internal Memory Free')
        else:
            print(atmcd_errors.Error_Codes(ret))


class AcquisitionThread(threading.Thread):
    running = False
    lock = threading.Lock()

    def __init__(self, cam):
        threading.Thread.__init__(self)
        self.cam = cam

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                self.cam.get_images()

    def stop(self):
        self.running = False
        self.join()


class DataList:

    def __init__(self, max_length):
        self.data_list = deque(maxlen=max_length)
        self.ind_list = deque(maxlen=max_length)

    def add_element(self, elements, start_ind, end_ind):
        self.data_list.extend(elements)
        self.ind_list.extend(list(range(start_ind, end_ind + 1)))

    def get_elements(self):
        return list(self.data_list)

    def get_last_element(self):
        return self.data_list[-1] if self.data_list else None

    def is_empty(self):
        return len(self.data_list) == 0
