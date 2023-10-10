import ctypes as ct
import sys

sys.path.append(r'C:\Program Files\Mad City Labs\NanoDrive')

nano_dll = r'C:\Program Files\Mad City Labs\NanoDrive\Madlib.dll'


class MCLNanoDrive:

    def __init__(self, logg=None):
        super().__init__()

        if logg is None:
            import logging
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
            self.logg = logging
        else:
            self.logg = logg

        self.mclpiezo = ct.cdll.LoadLibrary(nano_dll)

        self.handle = self.mclpiezo.MCL_InitHandle()
        if self.handle == 0:
            print("Error: self.handle not initialized correctly")
            return

        # find some basic information about the NanoDrive
        class ProductInfo(ct.Structure):
            _fields_ = [("axis_bitmap", ct.c_ubyte),
                        ("ADC_resolution", ct.c_short),
                        ("DAC_resolution", ct.c_short),
                        ("Product_id", ct.c_short),
                        ("FirmwareVersion", ct.c_short),
                        ("FirmwareProfile", ct.c_short)]
            _pack_ = 1  # this is how it is packed in the Madlib dll

        pi = ProductInfo()
        ppi = ct.pointer(pi)
        err = self.mclpiezo.MCL_GetProductInfo(ppi, self.handle)
        if err != 0:
            print("Error: NanoDrive could not get productInformation. Error Code:", err)
            self.mclpiezo.MCL_ReleaseHandle(self.handle)  # be sure to release self.handle anytime before returning
            return
        else:
            print("Information about the NanoDrive:")
            print("axis bitmap:", pi.axis_bitmap)
            print("ADC resolution:", pi.ADC_resolution)
            print("DAC resolution:", pi.DAC_resolution)
            print("Product ID:", pi.Product_id)
            print("Firmware Version:", pi.FirmwareVersion)
            print("Firmware Profile:", pi.FirmwareProfile)

        self.axis = []
        if (pi.axis_bitmap & 0x1) == 0x1:
            self.axis.append(ct.c_uint(1))
            print("Using X axis")
        if (pi.axis_bitmap & 0x2) == 0x2:
            self.axis.append(ct.c_uint(2))
            print("Using Y axis")
        if (pi.axis_bitmap & 0x4) == 0x4:
            self.axis.append(ct.c_uint(3))
            print("Using Z axis")
        else:
            print("No valid axes available")
            self.mclpiezo.MCL_ReleaseHandle(self.handle)
            return

        # when function returns a c-type that is not an integer, must set the return type before you ever use it
        cal = self.mclpiezo.MCL_GetCalibration
        cal.restype = ct.c_double
        readpos = self.mclpiezo.MCL_SingleReadN
        readpos.restype = ct.c_double

        self.calibration = []
        for i in range(len(self.axis)):
            self.calibration.append(self.mclpiezo.MCL_GetCalibration(self.axis[i], self.handle))

    def __del__(self):
        pass

    def close(self):
        """
        Closes the connection by releasing the handle.
        """
        # for i in range(len(self.axis)):
        #     print(self.move_position(i, 0.0))
        self.mclpiezo.MCL_ReleaseHandle(self.handle)
        print("Piezo Handle released")

    def read_position(self, ax):
        pos = self.mclpiezo.MCL_SingleReadN(self.axis[ax], self.handle)
        if pos < 0:
            print("Error: NanoDrive did not correctly read position. Error Code:", pos)
            self.mclpiezo.MCL_ReleaseHandle(self.handle)
            return
        else:
            print("Current position is", pos)
            return pos

    def move_position(self, ax, pos):
        err = self.mclpiezo.MCL_SingleWriteN(ct.c_double(pos), self.axis[ax], self.handle)
        if err != 0:
            print("Error: NanoDrive did not correctly write position. Error Code:", err)
            self.mclpiezo.MCL_ReleaseHandle(self.handle)
            return
        else:
            # pause before reading again
            self.mclpiezo.MCL_DeviceAttached(100, self.handle)
            # read the new position to make sure it actually moved
            pos = self.mclpiezo.MCL_SingleReadN(self.axis[ax], self.handle)
            if pos < 0:
                print("Error: NanoDrive did not correctly read position. Error Code:", pos)
                self.mclpiezo.MCL_ReleaseHandle(self.handle)
                return
            else:
                return pos
