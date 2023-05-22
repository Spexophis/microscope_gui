import sys

sys.path.append(r'C:\Program Files\The Imaging Source Europe GmbH\sources')

import ctypes
import numpy as np
import tisgrabber as tis

ic = ctypes.cdll.LoadLibrary(r'C:\Program Files\The Imaging Source Europe GmbH\sources\tisgrabber_x64.dll')
tis.declareFunctions(ic)
ic.IC_InitLibrary(0)


class CallbackUserdata(ctypes.Structure):
    """ Example for user data passed to the callback function."""

    def __init__(self, ):
        self.unsused = ""


def frameReadyCallback(hGrabber, pBuffer, framenumber, pData):
    Width = ctypes.c_long()
    Height = ctypes.c_long()
    BitsPerPixel = ctypes.c_int()
    colorformat = ctypes.c_int()

    # Query the image description values
    ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel,
                              colorformat)

    # Calculate the buffer size
    bpp = int(BitsPerPixel.value / 8.0)
    buffer_size = Width.value * Height.value * bpp

    if buffer_size > 0:
        image = ctypes.cast(pBuffer,
                            ctypes.POINTER(
                                ctypes.c_ubyte * buffer_size))

        cvMat = np.ndarray(buffer=image.contents,
                           dtype=np.uint8,
                           shape=(Height.value,
                                  Width.value,
                                  bpp))


frameReadyCallbackfunc = ic.FRAMEREADYCALLBACK(frameReadyCallback)
userdata = CallbackUserdata()


class TISCamera:

    def __init__(self):
        super().__init__()

        device_count = ic.IC_GetDeviceCount()
        if device_count == 1:
            print("Device {}".format(tis.D(ic.IC_GetDevice(0))))
            self.unique_name = tis.D(ic.IC_GetUniqueNamefromList(0))
            print("Unique Name : {}".format(self.unique_name))
            self.hGrabber = ic.IC_CreateGrabber()
            if ic.IC_OpenDevByUniqueName(self.hGrabber, tis.T(self.unique_name)) == tis.IC_SUCCESS:
                print("SUCCESS: TIS Camera ON")
                if ic.IC_IsDevValid(self.hGrabber):
                    if ic.IC_SetPropertyAbsoluteValue(self.hGrabber, tis.T("Gain"), tis.T("Value"),
                                                      ctypes.c_float(0)) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Gain zero")
                    else:
                        print("FAIL: Set Gain zero")
                    if ic.IC_SetPropertyValue(self.hGrabber, tis.T("Brightness"), tis.T("Value"),
                                              ctypes.c_int(0)) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Brightness zero")
                    else:
                        print("FAIL: Set Brightness zero")
                    if ic.IC_SetContinuousMode(self.hGrabber, 1) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Continuous Mode")
                    else:
                        print("FAIL: Set Continuous Mode")
                    if ic.IC_SetVideoFormat(self.hGrabber, tis.T("Y800 (2448x2048)")) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Video Format Y800 (2448x2048)")
                    else:
                        print("FAIL: Set Video Format Y800 (2448x2048)")
                    if ic.IC_SetFrameRate(self.hGrabber, ctypes.c_float(30.0)) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Frame Rate 30fps")
                    else:
                        print("FAIL: Set Frame Rate 30fps")
                    self.data = None
                else:
                    print("Invalid TISGrabber")
            else:
                print("FAIL: TIS Camera ON")
        else:
            print('No TIS camera')

    def close(self):
        r = ic.IC_CloseVideoCaptureDevice(self.hGrabber)
        if r:
            r = ic.IC_ReleaseGrabber(self.hGrabber)
            if r:
                ic.IC_CloseLibrary()
                print("TIS Camera OFF")

    def prepare_live(self):
        pass

    def start_live(self):
        if ic.IC_IsDevValid(self.hGrabber):
            ic.IC_StartLive(self.hGrabber, 0)

    def suspend_live(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            ic.IC_SuspendLive(self.hGrabber)

    def stop_live(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            ic.IC_StopLive(self.hGrabber)

    def set_trigger_mode(self):
        if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("Enable"), 1) == tis.IC_SUCCESS:
            print("SUCCESS: Set Trigger Enable")
        else:
            print("FAIL: Set Trigger Enable")
        if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("IMX Low-Latency Mode"),
                                   1) == tis.IC_SUCCESS:
            print("SUCCESS: Set Trigger IMX Low-Latency Mode")
        else:
            print("FAIL: Set Trigger IMX Low-Latency Mode")
        if ic.IC_SetPropertyMapString(self.hGrabber, tis.T("Trigger"), tis.T("Exposure Mode"),
                                      tis.T("Timed")) == tis.IC_SUCCESS:
            print("SUCCESS: Set Trigger Exposure Mode Timed")
        else:
            print("FAIL: Set Trigger Exposure Mode Timed")
        if ic.IC_SetPropertyMapString(self.hGrabber, tis.T("Trigger"), tis.T("Exposure Mode"),
                                      tis.T("Trigger Width")) == tis.IC_SUCCESS:
            print("SUCCESS: Set Trigger Exposure Mode Trigger Width")
        else:
            print("FAIL: Set Trigger Exposure Mode Trigger Width")
        if ic.IC_RemoveOverlay(self.hGrabber, 1) == tis.IC_SUCCESS:
            print("SUCCESS: Remove Overlay")
        else:
            print("FAIL: Remove Overlay")

    def get_gain(self, verbose=True):
        if ic.IC_IsDevValid(self.hGrabber):
            gain_min = ctypes.c_long()
            gain_max = ctypes.c_long()
            gain = ctypes.c_long()
            ic.IC_GetPropertyValue(self.hGrabber, tis.T("Gain"), tis.T("Value"), gain)
            ic.IC_GetPropertyValueRange(self.hGrabber, tis.T("Gain"), tis.T("Value"),
                                        gain_min, gain_max)
            print("Gain is {0} range is {1} - {2}".format(gain.value, gain_min.value, gain_max.value))
            if verbose:
                return gain.value

    def set_gain(self, gain):
        if ic.IC_IsDevValid(self.hGrabber):
            r = ic.IC_SetPropertyAbsoluteValue(self.hGrabber, tis.T("Gain"), tis.T("Value"), ctypes.c_float(gain))
            if r:
                self.get_gain(False)

    def get_exposure(self, verbose=True):
        if ic.IC_IsDevValid(self.hGrabber):
            expo_min = ctypes.c_float()
            expo_max = ctypes.c_float()
            exposure = ctypes.c_float()
            ic.IC_GetPropertyAbsoluteValue(self.hGrabber, tis.T("Exposure"), tis.T("Value"), exposure)
            ic.IC_GetPropertyAbsoluteValueRange(self.hGrabber, tis.T("Exposure"), tis.T("Value"), expo_min, expo_max)
            print("Exposure is {0}, range is {1} - {2}".format(exposure.value, expo_min.value, expo_max.value))
            if verbose:
                return exposure.value

    def set_exposure(self, exposure):
        if ic.IC_IsDevValid(self.hGrabber):
            r = ic.IC_SetPropertyAbsoluteValue(self.hGrabber, tis.T("Exposure"), tis.T("Value"),
                                               ctypes.c_float(exposure))
            if r:
                self.get_exposure(False)

    def get_last_image(self):
        self.snap_image()
        self.data = self.get_data()

    def snap_image(self):
        if ic.IC_SnapImage(self.hGrabber, 2000) == tis.IC_SUCCESS:
            print("SUCCESS: Image Capture")
            return True
        else:
            print("FAIL: Image Capture")
            return False

    def get_buffer(self):
        img_width = ctypes.c_long()
        img_height = ctypes.c_long()
        img_depth = ctypes.c_int()
        color_format = ctypes.c_int()
        if ic.IC_GetImageDescription(self.hGrabber, img_width, img_height, img_depth, color_format) == tis.IC_SUCCESS:
            bpp = int(img_depth.value / 8.0)
            buffer_size = img_width.value * img_height.value * img_depth.value
            return bpp, buffer_size, img_width, img_height, img_depth

    def get_data(self):
        bpp, buffer_size, width, height, depth = self.get_buffer()
        image_pointer = ic.IC_GetImagePtr(self.hGrabber)
        imagedata = ctypes.cast(image_pointer, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
        image = np.ndarray(buffer=imagedata.contents, dtype=np.uint16, shape=(height.value, width.value, bpp))
        return image
