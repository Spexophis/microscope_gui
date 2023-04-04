import numpy as np
from pyicic import IC_ImagingControl


class TISCamera:

    def __init__(self):
        super().__init__()

        ic_ic = IC_ImagingControl.IC_ImagingControl()
        ic_ic.init_library()
        cam_names = ic_ic.get_unique_device_names()

        if cam_names:
            print(cam_names[0])
            self.model = cam_names[0]
            self.cam = ic_ic.get_device(cam_names[0])

            self.cam.open()

            self.shape = (0, 0)
            self.cam.colorenable = 0

            self.cam.enable_continuous_mode(True)  # image in continuous mode
            self.cam.enable_trigger(False)  # camera will wait for trigger
            self.formats = self.cam.list_video_formats()
            self.cam.set_video_format(self.formats[39])

            self.handle = True
        else:
            self.handle = False
            print('No TISCamera')

    def close(self):
        if self.handle:
            self.cam.close()

    def start_live(self):
        self.cam.start_live()  # start imaging

    def stop_live(self):
        self.cam.stop_live()  # stop imaging

    def suspend_live(self):
        self.cam.suspend_live()  # suspend imaging into prepared state

    def prepare_live(self):
        self.cam.prepare_live()  # prepare prepared state for live imaging

    def reset_frame_state(self):
        self.cam.reset_frame_ready()

    def wait_for_frame(self, timeout=1000):
        self.cam.wait_til_frame_ready(timeout)

    def grabFrame(self):
        # self.cam.wait_til_frame_ready(100)  # wait for frame ready
        frame, width, height, depth = self.cam.get_image_data()
        frame = np.array(frame, dtype='float64')
        # Check if below is giving the right dimensions out
        # TODO: do this smarter, as I can just take every 3rd value instead of creating a reshaped
        #       3D array and taking the first plane of that
        frame = np.reshape(frame, (height, width, depth))[:, :, 0]
        self.frame = np.transpose(frame)
        # print('Image frame grabbed successfully')
        # tf.imsave('test.tif', self.frame)

        return self.frame

    def setROI(self, hpos, vpos, hsize, vsize):
        hsize = max(hsize, 256)  # minimum ROI size
        vsize = max(vsize, 24)  # minimum ROI size
        # self.cam.frame_filter_set_parameter(self.roi_filter, 'Top'.encode('utf-8'), vpos)
        self.cam.frame_filter_set_parameter(self.roi_filter, 'Top', vpos)
        self.cam.frame_filter_set_parameter(self.roi_filter, 'Left', hpos)
        self.cam.frame_filter_set_parameter(self.roi_filter, 'Height', vsize)
        self.cam.frame_filter_set_parameter(self.roi_filter, 'Width', hsize)
        # top = self.cam.frame_filter_get_parameter(self.roi_filter, 'Top')
        # left = self.cam.frame_filter_get_parameter(self.roi_filter, 'Left')
        # hei = self.cam.frame_filter_get_parameter(self.roi_filter, 'Height')
        # wid = self.cam.frame_filter_get_parameter(self.roi_filter, 'Width')

    def setPropertyValue(self, property_name, property_value):
        # Check if the property exists.
        if property_name == "gain":
            self.cam.gain = property_value
        elif property_name == "brightness":
            self.cam.brightness = property_value
        elif property_name == "exposure":
            self.cam.exposure = property_value
        elif property_name == 'image_height':
            self.shape = (self.shape[0], property_value)
        elif property_name == 'image_width':
            self.shape = (property_value, self.shape[1])
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def getPropertyValue(self, property_name):
        # Check if the property exists.
        if property_name == "gain":
            property_value = self.cam.gain.value
        elif property_name == "brightness":
            property_value = self.cam.brightness.value
        elif property_name == "exposure":
            property_value = self.cam.exposure.value
        elif property_name == "image_width":
            property_value = self.shape[0]
        elif property_name == "image_height":
            property_value = self.shape[1]
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def openPropertiesGUI(self):
        self.cam.show_property_dialog()
