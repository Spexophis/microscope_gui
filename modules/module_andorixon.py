import sys
import threading
from collections import deque
from typing import Optional

import numpy as np
from pyAndorSDK2 import atmcd, atmcd_errors

sys.path.append(r'C:\Program Files\Andor SDK')

Readout_Mode = {0: "Full Vertical Binning", 1: "Multi-Track", 2: "Random-Track", 3: "Single-Track", 4: "Image"}
Trigger_Mode = {0: "Internal", 1: "External", 6: "External Start", 7: "External Exposure", 10: "Software"}
Acquisition_Mode = {1: "Single Scan", 2: "Accumulate", 3: "Kinetics", 4: "Fast Kinetics", 5: "Run Till Abort"}


class EMCCDCamera:
    class CameraSettings:
        def __init__(self):
            self.temperature = None
            self.gain = 0
            self.t_clean = None
            self.t_readout = None
            self.t_exposure = None
            self.t_accumulate = None
            self.t_kinetic = None
            self.bin_h = 1
            self.bin_v = 1
            self.start_h = 1
            self.end_h = 1024
            self.start_v = 1
            self.end_v = 1024
            self.pixels_x = 1024
            self.pixels_y = 1024
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 13  # micron
            self.buffer_size = None
            self.acq_num = 0
            self.acq_first = 0
            self.acq_last = 0
            self.valid_index = 0

    def __init__(self, logg=None):
        if logg is None:
            import logging
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
            self.logg = logging
        else:
            self.logg = logg
        self._settings = self.CameraSettings()
        self.sdk = self._initialize_sdk()
        if self.sdk:
            self._configure_camera()
        self.data = None
        self.acq_thread = None

    def __del__(self):
        pass

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def _initialize_sdk(self):
        try:
            sdk = atmcd(r'C:\Program Files\Andor SDK')
            ret = sdk.Initialize(r'C:/Program Files/Andor SDK/atmcd64d.dll')
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                return sdk
            else:
                self.logg.error('AndorEMCCD is not initiated')
                return None
        except Exception as e:
            self.logg.error(f"Error initializing SDK: {e}")
            return None

    def _configure_camera(self):
        try:
            self.get_sn()
            self.cooler_on()
            self.set_frame_transfer(0)
            self.set_readout_rate(0, 3)
        except Exception as e:
            self.logg.error(f"Error configuring camera: {e}")

    def close(self):
        self.cooler_off()
        ret = self.sdk.ShutDown()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Andor EMCCD Shut Down")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_sn(self):
        ret, serial_number = self.sdk.GetCameraSerialNumber()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(f"Camera Serial Number : {serial_number}")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def cooler_on(self):
        ret = self.sdk.CoolerON()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("EMCCD Cooler ON")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def cooler_off(self):
        ret = self.sdk.CoolerOFF()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("EMCCD Cooler OFF")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_ccd_temperature(self):
        ret, self.temperature = self.sdk.GetTemperature()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("EMCCD Temperature is {}".format(self.temperature))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def check_camera_status(self):
        ret, status = self.sdk.GetStatus()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(atmcd_errors.Error_Codes(status))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_sensor_size(self):
        ret, self.pixels_x, self.pixels_y = self.sdk.GetDetector()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.img_size = self.pixels_x * self.pixels_y
            self.logg.info("Detector size: pixels_x = {} pixels_y = {}".format(self.pixels_x, self.pixels_y))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_readout_mode(self, ind):
        """
        0 - Full Vertical Binning
        1 - Multi-Track
        2 - Random-Track
        3 - Single-Track
        4 - Image
        """
        ret = self.sdk.SetReadMode(ind)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Readout Mode to {}".format(Readout_Mode[ind]))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_readout_rate(self, hs=0, vs=3):
        ret = self.sdk.SetHSSpeed(0, hs)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Horizontal Speed")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        ret = self.sdk.SetVSSpeed(vs)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Vertical Speed")
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_frame_transfer(self, ft=1):
        ret = self.sdk.SetFrameTransferMode(ft)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Frame Transfer Mode {}".format(ft))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_gain(self):
        ret = self.sdk.SetEMCCDGain(self.gain)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.get_gain()
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_gain(self):
        ret, self.gain = self.sdk.GetEMCCDGain()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("CCD EMGain is {}".format(self.gain))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_roi(self):
        ret = self.sdk.SetImage(self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("bin_h = {} \nbin_v = {} \nstart_h = {} \nend_h = {} \nstart_v = {} \nend_v = {}".format(
                self.bin_h, self.bin_v, self.start_h, self.end_h, self.start_v, self.end_v))
            self.pixels_x = self.end_h - self.start_h + 1
            self.pixels_y = self.end_v - self.start_v + 1
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 13 / self.bin_h
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_trigger_mode(self, ind):
        """
        0 - Internal
        1 - External
        6 - External Start
        7 - External Exposure
        10 - Software
        """
        ret = self.sdk.SetTriggerMode(ind)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Trigger Mode Set to {}".format(Trigger_Mode[ind]))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_exposure_time(self):
        ret = self.sdk.SetExposureTime(self.t_exposure)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Exposure Time to {}".format(self.t_exposure))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_acquisition_mode(self, ind):
        """
        1 - Single Scan
        2 - Accumulate
        3 - Kinetics
        4 - Fast Kinetics
        5 - Run Till Abort
        """
        ret = self.sdk.SetAcquisitionMode(ind)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Acquisition Mode to {}".format(Acquisition_Mode[ind]))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_acquisition_timings(self):
        ret, self.t_exposure, self.t_accumulate, self.t_kinetic = self.sdk.GetAcquisitionTimings()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Acquisition Timings exposure = {} accumulate = {} kinetic = {}".format(self.t_exposure,
                                                                                                       self.t_accumulate,
                                                                                                       self.t_kinetic))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        ret, self.t_readout = self.sdk.GetReadOutTime()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Readout Time = {}".format(self.t_readout))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))
        ret, self.t_clean = self.sdk.GetKeepCleanTime()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Keep Clean Time = {}".format(self.t_clean))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_buffer_size(self):
        ret, self.buffer_size = self.sdk.GetSizeOfCircularBuffer()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Get Circular Buffer = {}".format(self.buffer_size))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_kinetic_cycle_time(self, t):
        ret = self.sdk.SetKineticCycleTime(t)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Kinetic Cycle Time to {}".format(t))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def set_kinetics_num(self, kn):
        ret = self.sdk.SetNumberKinetics(kn)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info("Set Number of Kinetics to {}".format(kn))
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

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
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.acq_thread.start()
            self.logg.info('Start live image')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def stop_live(self):
        self.acq_thread.stop()
        self.acq_thread = None
        ret = self.sdk.AbortAcquisition()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Live image stopped')
            self.data = None
            self.free_memory()
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_images(self):
        ret, first, last = self.sdk.GetNumberNewImages()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            num = last - first + 1
            (ret, data_array, valid_first, valid_last) = self.sdk.GetImages16(first, last, self.img_size * num)
            if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
                self.valid_index = valid_last
                data_array = np.split(data_array.reshape(num, self.img_size), num, axis=0)
                data_array = [subarray.reshape(self.pixels_x, self.pixels_y) for subarray in data_array]
                self.data.add_element(data_array, valid_first, valid_last)
            # else:
            #     self.logg.error(atmcd_errors.Error_Codes(ret))
        # else:
        #     self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_last_image(self) -> np.ndarray:
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
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Ready to acquire data')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def start_data_acquisition(self):
        ret = self.sdk.StartAcquisition()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Kinetic acquisition start')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def get_acq_num(self):
        ret, first, last = self.sdk.GetNumberAvailableImages()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.acq_first, self.acq_last = first, last
            self.logg.info(first, last)
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def check_acquisition_progress(self):
        ret, self.numAccumulate, self.numKinetics = self.sdk.GetAcquisitionProgress()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info(
                "GetAcquisitionProgress returned {} \n"
                "number of accumulations completed = {} \n"
                "kinetic scans completed = {}".format(ret, self.numAccumulate, self.numKinetics))

    def get_data(self, num) -> Optional[np.ndarray]:
        ret, data_array = self.sdk.GetAcquiredData16(num * self.img_size)
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Data Retrieved')
            return data_array.reshape(num, self.pixels_x, self.pixels_y)
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))

    def free_memory(self):
        ret = self.sdk.FreeInternalMemory()
        if ret == atmcd_errors.Error_Codes.DRV_SUCCESS:
            self.logg.info('Internal Memory Free')
        else:
            self.logg.error(atmcd_errors.Error_Codes(ret))


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

    def get_elements(self) -> list:
        return list(self.data_list)

    def get_last_element(self) -> Optional[np.ndarray]:
        return self.data_list[-1] if self.data_list else None

    def is_empty(self) -> bool:
        return len(self.data_list) == 0
