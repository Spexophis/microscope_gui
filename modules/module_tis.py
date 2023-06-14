import sys

sys.path.append(r'C:\Program Files\The Imaging Source Europe GmbH\sources')

import ctypes
import numpy as np
import tisgrabber as tis

ic = ctypes.cdll.LoadLibrary(r'C:\Program Files\The Imaging Source Europe GmbH\sources\tisgrabber_x64.dll')
tis.declareFunctions(ic)
ic.IC_InitLibrary(0)


class CallbackUserdata(ctypes.Structure):
    def __init__(self):
        super().__init__()
        self.index = 1


def Frame_Callback(hGrabber, pBuffer, framenumber, pData):
    print("Callback called", pData.index)
    pData.index = pData.index + 1


Userdata = CallbackUserdata()
Callback_Function = ic.FRAMEREADYCALLBACK(Frame_Callback)


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
                    if ic.IC_SetPropertyValue(self.hGrabber, tis.T("Brightness"), tis.T("Value"),
                                              ctypes.c_int(0)) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Brightness zero")
                    else:
                        print("FAIL: Set Brightness zero")
                    if ic.IC_SetContinuousMode(self.hGrabber, 0) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Continuous Mode ON")
                    else:
                        print("FAIL: Set Continuous Mode ON")
                    if ic.IC_SetFormat(self.hGrabber, tis.SinkFormats.Y16.value) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Format Y16")
                    else:
                        print("FAIL: Set Format Y16")
                    if ic.IC_SetVideoFormat(self.hGrabber, tis.T("Y16 (2448x2048)")) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Video Format Y16 (2448x2048)")
                    else:
                        print("FAIL: Set Video Format Y16 (2448x2048)")
                    if ic.IC_SetFrameRate(self.hGrabber, ctypes.c_float(37.5)) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Frame Rate 37.5fps")
                    else:
                        print("FAIL: Set Frame Rate 37.5fps")
                    if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Gain"), tis.T("Auto"), 1) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Auto Gain")
                    else:
                        print("FAIL: Set Auto Gain")
                    if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Exposure"), tis.T("Auto"), 1) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Exposure Auto")
                    else:
                        print("FAIL: Set Exposure Auto")
                    if ic.IC_SetPropertyValue(self.hGrabber, tis.T("Denoise"), tis.T("Value"),
                                              ctypes.c_int(0)) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Denoise to 0")
                    else:
                        print("FAIL: Set Denoise")
                    if ic.IC_SetFrameReadyCallback(self.hGrabber, Callback_Function, Userdata) == tis.IC_SUCCESS:
                        print("SUCCESS: Set Frame Ready Callback")
                    else:
                        print("FAIL: Set Frame Ready Callback")
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
        if ic.IC_PrepareLive(self.hGrabber, 0):
            print('Live ready')

    def start_live(self):
        if ic.IC_IsDevValid(self.hGrabber):
            if ic.IC_StartLive(self.hGrabber, 0) == tis.IC_SUCCESS:
                print('Live start')

    def suspend_live(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            ic.IC_SuspendLive(self.hGrabber)

    def stop_live(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            ic.IC_StopLive(self.hGrabber)
            print('Live stop')

    def send_trigger(self):
        if ic.IC_IsDevValid(self.hGrabber) & ic.IC_IsLive(self.hGrabber):
            # ic.IC_PropertyOnePush(self.hGrabber, tis.T("Trigger"), tis.T("Software Trigger"))
            ic.IC_SoftwareTrigger(self.hGrabber)
            print('Software Trigger')

    def set_trigger_mode(self, sw=1):
        if sw:
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Gain"), tis.T("Auto"), 0) == tis.IC_SUCCESS:
                print("SUCCESS: Set Auto Gain OFF")
            else:
                print("FAIL: Set Auto Gain OFF")
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Exposure"), tis.T("Auto"), 0) == tis.IC_SUCCESS:
                print("SUCCESS: Set Exposure Auto OFF")
            else:
                print("FAIL: Set Exposure Auto OFF")
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("Enable"), 1) == tis.IC_SUCCESS:
                print("SUCCESS: Set Trigger Enable")
            else:
                print("FAIL: Set Trigger Enable")
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("IMX Low-Latency Mode"),
                                       1) == tis.IC_SUCCESS:
                print("SUCCESS: Set Trigger IMX Low-Latency Mode")
            else:
                print("FAIL: Set Trigger IMX Low-Latency Mode")
            # if ic.IC_SetPropertyMapString(self.hGrabber, tis.T("Trigger"), tis.T("Exposure Mode"),
            #                               tis.T("Timed")) == tis.IC_SUCCESS:
            #     print("SUCCESS: Set Trigger Exposure Mode Timed")
            # else:
            #     print("FAIL: Set Trigger Exposure Mode Timed")
            if ic.IC_SetPropertyMapString(self.hGrabber, tis.T("Trigger"), tis.T("Exposure Mode"),
                                          tis.T("Trigger Width")) == tis.IC_SUCCESS:
                print("SUCCESS: Set Trigger Exposure Mode Trigger Width")
            else:
                print("FAIL: Set Trigger Exposure Mode Trigger Width")
            if ic.IC_RemoveOverlay(self.hGrabber, 1) == tis.IC_SUCCESS:
                print("SUCCESS: Remove Overlay")
            else:
                print("FAIL: Remove Overlay")
        else:
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Gain"), tis.T("Auto"), 1) == tis.IC_SUCCESS:
                print("SUCCESS: Set Auto Gain")
            else:
                print("FAIL: Set Auto Gain")
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Exposure"), tis.T("Auto"), 1) == tis.IC_SUCCESS:
                print("SUCCESS: Set Exposure Auto")
            else:
                print("FAIL: Set Exposure Auto")
            if ic.IC_SetPropertySwitch(self.hGrabber, tis.T("Trigger"), tis.T("Enable"), 0) == tis.IC_SUCCESS:
                print("SUCCESS: Set Trigger Disable")
            else:
                print("FAIL: Set Trigger Disable")

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

    # def get_last_image(self):
    #     if self.snap_image():
    #         avg = np.float32(self.get_data())
    #         for i in range(16):
    #             if self.snap_image():
    #                 cv2.accumulateWeighted(np.float32(self.get_data()), avg, 0.2)
    #             else:
    #                 pass
    #         return avg
    #     else:
    #         pass

    def get_last_image(self):
        self.data = []
        if self.snap_image():
            self.data.append(self.get_data())
            for i in range(16):
                if self.snap_image():
                    self.data.append(self.get_data())
                else:
                    pass
            self.data = np.asarray(self.data)
            self.data = np.mean(self.data, axis=0)
            return self.data
        else:
            pass

    def snap_image(self):
        if ic.IC_SnapImage(self.hGrabber, 2000) == tis.IC_SUCCESS:
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
            img_depth = int(img_depth.value / 16.0) * ctypes.sizeof(ctypes.c_uint16)
            buffer_size = img_width.value * img_height.value * img_depth
            return buffer_size, img_width, img_height, img_depth
        else:
            print("FAIL: Get Buffer Size")

    def get_data(self):
        buffer_size, width, height, depth = self.get_buffer()
        image_pointer = ic.IC_GetImagePtr(self.hGrabber)
        image_data = ctypes.cast(image_pointer, ctypes.POINTER(ctypes.c_ubyte * int(buffer_size)))
        image = np.ndarray(shape=(height.value, width.value), buffer=image_data.contents, dtype=np.uint16)
        return image

    def show_property(self):
        ic.IC_ShowPropertyDialog(self.hGrabber)

    def save_img(self):
        ic.IC_SaveImage(self.hGrabber, tis.T(r'C:\Users\ruizhe.lin\Desktop\test.jpg'), tis.ImageFileTypes['JPEG'], 100)
