import pylablib as pll

pll.par["devices/dlls/uc480"] = r'C:\Program Files\Thorlabs\Scientific Imaging\ThorCam'

from pylablib.devices import uc480


class UC480Cam():

    def __init__(self):
        super().__init__()

        cam_info = uc480.list_cameras()
        if cam_info:
            print(cam_info)
            self.ucam = uc480.UC480Camera(cam_id=1)
            self.ucam.set_gain_boost(enabled=False)
            self.ucam.set_gains(None, None, None, None)
            # self.ucam.set_frame_format("array")
            self.handle = True
        else:
            self.handle = False
            print('No UC480Camera')

    def close(self):
        if self.handle:
            if self.ucam.acquisition_in_progress():
                self.ucam.stop_acquisition()
            self.ucam.clear_acquisition()
            self.ucam.close()

    def start_acquire(self):
        if self.ucam.acquisition_in_progress():
            self.ucam.stop_acquisition()
        else:
            self.ucam.clear_acquisition()
            self.ucam.start_acquisition()

    def stop_acquire(self):
        if self.ucam.acquisition_in_progress():
            self.ucam.stop_acquisition()

    def clear_acquire(self):
        self.ucam.clear_acquisition()

    def set_exposure(self, expo):
        result = self.ucam.set_exposure(expo)
        print("Exposre time set to {}".format(result))

    def get_last_image(self):
        self.img = self.ucam.read_newest_image()
        return self.img

    def get_image_stack(self):
        self.imgstack = self.ucam.read_multiple_images()
        return self.imgstack

    def snap_image(self):
        self.img = self.ucam.snap()
        return self.img
