import ctypes as ct
import copy
import numpy as np
from PIL import Image


class HamamatsuSLM:

    def __init__(self):
        super().__init__()

        self.lcos_lib = ct.cdll.LoadLibrary("Image_Control.dll")
        self.pitch = 1  # pixel pitch (0: 20um 1: 1.25um)
        # SLM pixel numbers
        self.x = 1272
        self.y = 1024
        self.array_size = self.x * self.y
        # make the 8bit unsigned integer array type
        self.FARRAY = ct.c_uint8 * self.array_size
        self.farray = self.FARRAY(0)
        self.farray2 = self.FARRAY(0)
        self.farray3 = self.FARRAY(0)
        # monitor number setting
        self.monitorNo = 2
        self.windowNo = 0
        self.xShift = 0
        self.yShift = 0

    def display_axiconLens_pattern(self, top=10.0):
        self.make_axicon_lens(top, self.pitch, self.x, self.y, self.farray)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray)

    def display_cylindricalLens_pattern(self, focus=1000, wavelength=488, modeSelect=0):
        self.make_cylindrical_lens(focus, wavelength, self.pitch, modeSelect, self.x, self.y, self.farray)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray)

    def display_rotated_pattern(self, degree=30.0):
        self.image_rotation(self.farray, degree, self.x, self.y, self.farray2)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray2)

    def display_diffraction_grating_pattern(self, rowOrColumn=0, gradiationNo=16, gradiationWidth=16, slipFactor=0):
        self.make_diffraction_pattern(rowOrColumn, gradiationNo, gradiationWidth, slipFactor, self.x, self.y,
                                      self.farray)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray)

    def display_Laguerre_Gauss_mode_pattern(self, p=1, m=1, pitch=1, beamSize=20.0):
        self.make_laguerre_gaussian(p, m, pitch, beamSize, self.x, self.y, self.farray)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray)

    def display_FresnelLens_pattern(self, focus=1000, wavelength=1064):
        self.make_fresnel_lens(focus, wavelength, self.pitch, self.x, self.y, self.farray2)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray2)

    def display_synthesize_FresnelLens_Laguerre_Gauss_Mode_pattern(self):
        self.phase_synthsizer([self.farray, self.farray2], self.farray3)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray3)

    def display_CGH_pattern(self, filepath="Target image sample/char_hpk_128x128.bmp"):
        self.make_bmp_array(filepath, self.x, self.y, self.farray)
        self.show_on_display(self.monitorNo, self.windowNo, self.x, self.xShift, self.y, self.yShift, self.farray)

    def make_axicon_lens(self, top, pitch, x, y, array):
        """
        the function for making AxiconLens pattern array
        double top: Top level of AxiconLens pattern. (/pi rad)
        int pitch: Pixel pitch. 0: 20um 1: 1.25um
        int x: Pixel number of x-dimension
        int y: Pixel number of y-dimension
        8bit unsigned integer array: output array
        """
        AxiconLens = self.lcos_lib.AxiconLens
        AxiconLens.argtyes = [ct.c_double, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        AxiconLens.restype = ct.c_int
        if (pitch != 0 and pitch != 1):
            print("Error: AxiconLensFunction. invalid argument (pitch).")
            return -1
        # input argument to dll function.
        AxiconLens(ct.c_double(top), pitch, x, y, ct.byref(ct.c_int(x * y)), ct.byref(array))

    def make_cylindrical_lens(self, focus, wavelength, pitch, modeSelect, x, y, array):
        """
        the function for making CylindricalLens pattern array
        int focus: the forcus of cylindrical lens. (mm)
        int wavelength: the wavelength of light. (nm)
        int pitch: Pixel pitch. 0: 20um 1: 1.25um
        int modeSelect: 0: horizontal or 1: vertical
        int x: Pixel number of x-dimension
        int y: Pixel number of y-dimension
        8bit unsigned int array: output array
        """
        CylindricalLens = self.lcos_lib.CylindricalLens
        CylindricalLens.argtyes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        CylindricalLens.restype = ct.c_int
        if (pitch != 0 and pitch != 1):
            print("Error: CylindricalLensFunction. invalid argument (pitch).")
            return -1
        CylindricalLens(focus, wavelength, pitch, modeSelect, x, y, ct.byref(ct.c_int(x * y)), ct.byref(array))

    def make_diffraction_pattern(self, rowOrColumn, gradiationNo, gradiationWidth, slipFactor, x, y, array):
        """
        the function for making Diffraction pattern array
        int rowOrColumn: 0: horizontal or 1: vertical
        int gradiationNo: the number of gradiation.
        int gradiationWidth: the width of gradiation.
        int slipFactor: slip factor.
        int x: Pixel number of x-dimension
        int y: Pixel number of y-dimension
        8bit unsigned int array array: output array
        """
        Diffraction_pattern = self.lcos_lib.Diffraction_pattern
        Diffraction_pattern.argtyes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        Diffraction_pattern.restype = ct.c_int
        Diffraction_pattern(rowOrColumn, gradiationNo, gradiationWidth, slipFactor, x, y, ct.byref(ct.c_int(x * y)),
                            ct.byref(array))

    def make_laguerre_gaussian(self, p, m, pitch, beamSize, x, y, array):
        """
        the function for making LaguerreGaussMode pattern array
        int p: radial index
        int m: azimuthal index
        int pitch: Pixel pitch. 0: 20um 1: 1.25um
        double beamSize: Beam size (mm)
        int x: Pixel number of x-dimension
        int y: Pixel number of y-dimension
        8bit unsigned int array array: output array
        """
        LaguerreGaussMode = self.lcos_lib.LaguerreGaussMode
        LaguerreGaussMode.argtyes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        LaguerreGaussMode.restype = ct.c_int
        if (pitch != 0 and pitch != 1):
            print("Error: LaguerreGaussModeFunction. invalid argument (pitch).")
            return -1
        LaguerreGaussMode(p, m, pitch, ct.c_double(beamSize), x, y, ct.byref(ct.c_int(x * y)), ct.byref(array))

    def make_fresnel_lens(self, forcus, wavelength, pitch, x, y, array):
        """
        the function for making FresnelLens pattern array
        int focus: the forcus of cylindrical lens. (mm)
        int wavelength: the wavelength of light. (nm)
        int pitch: Pixel pitch. 0: 20um 1: 1.25um
        int x: Pixel number of x-dimension
        int y: Pixel number of y-dimension
        8bit unsigned int array: output array
        """
        FresnelLens = self.lcos_lib.FresnelLens
        FresnelLens.argtyes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        FresnelLens.restype = ct.c_int
        if pitch != 0 and pitch != 1:
            print("Error: FresnelLensFunction. invalid argument (pitch).")
            return -1
        FresnelLens(forcus, wavelength, pitch, x, y, ct.byref(ct.c_int(x * y)), ct.byref(array))

    def make_bmp_array(self, filepath, x, y, outArray):
        """
        the function for making FresnelLens pattern array
        String filepath: image file path.
        int x: Pixel number of x-dimension
        int y: Pixel number of y-dimension
        8bit unsigned int array outArray: output array
        """
        im = Image.open(filepath)
        imageHeight, imageWidth = im.size
        im_gray = im.convert("L")
        print("Imagesize = {} x {}".format(imageWidth, imageHeight))
        for i in range(imageWidth):
            for j in range(imageHeight):
                outArray[i + imageWidth * j] = im_gray.getpixel((i, j))
        # Create CGH
        inArray = copy.deepcopy(outArray)
        Create_CGH_OC = self.lcos_lib.Create_CGH_OC
        Create_CGH_OC.argtyes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        Create_CGH_OC.restype = ct.c_int
        repNo = 100
        progressBar = 1
        Create_CGH_OC(ct.byref(inArray), repNo, progressBar, imageWidth, imageHeight,
                      ct.byref(ct.c_int(imageHeight * imageWidth)),
                      ct.byref(outArray))
        # Tilling the image
        inArray = copy.deepcopy(outArray)
        Image_Tiling = self.lcos_lib.Image_Tiling
        Image_Tiling.argtyes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        Image_Tiling.restype = ct.c_int
        Image_Tiling(ct.byref(inArray), imageWidth, imageHeight, imageHeight * imageWidth, x, y, ct.byref(ct.c_int(x * y)),
                     ct.byref(outArray))

    def show_on_display(self, monitorNo, windowNo, x, xShift, y, yShift, array):
        """
        the function for showing on LCOS display
        int monitorNo:
        int windowNo:
        int x: Pixel number of x-dimension
        int xShift: shift pixels of x-dimension
        int y: Pixel number of y-dimension
        int yShift: shift pixels of y-dimension
        8bit unsigned int array: output array
        """
        # Select LCOS window
        Window_Settings = self.lcos_lib.Window_Settings
        Window_Settings.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int]
        Window_Settings.restype = ct.c_int
        Window_Settings(monitorNo, windowNo, xShift, yShift)

        # Show pattern
        Window_Array_to_Display = self.lcos_lib.Window_Array_to_Display
        Window_Array_to_Display.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
        Window_Array_to_Display.restype = ct.c_int
        Window_Array_to_Display(array, x, y, windowNo, x * y)

    def stop_display(self, windowNo):
        Window_Term = self.lcos_lib.Window_Term
        Window_Term.argtyes = [ct.c_int]
        Window_Term.restype = ct.c_int
        Window_Term(windowNo)

    def image_rotation(self, inputArray, degree, x, y, outputArray):
        """
        the function for Srotating image
        input 1D array inputArray: input array.
        double degree: rotation degree (deg.)
        int x: Pixel number of x-dimension
        int y: Pixel number of y-dimension
        output 1D array outputArray: output array.
        """
        Image_Rotation = self.lcos_lib.Image_Rotation
        Image_Rotation.argtyes = [ct.c_void_p, ct.c_double, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        Image_Rotation.restype = ct.c_int
        Image_Rotation(ct.byref(inputArray), ct.c_double(degree), x, y, ct.byref(ct.c_int(x * y)), ct.byref(outputArray))

    def phase_synthsizer(self, inputPatterns, outputArray):
        """
        the function for Synthesizing image pattaerns
        input arrays 2D array inputPatterns: the compornents will be synthesized.
        output 1D array outputArray: output array.
        """
        n = len(inputPatterns[0])
        outputPattern = np.zeros(n, dtype=int)
        for pattern in inputPatterns:
            outputPattern = outputPattern + pattern
        for i in range(n):
            outputArray[i] = ct.c_uint8(outputPattern[i] % 256)
