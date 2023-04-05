from modules import module_andorixon
from modules import module_hamamatsu
# from modules import module_tiscamera
# from modules import module_thorcamera
from modules import module_deformablemirror
from modules import module_laser
from modules import module_daq
from modules import module_mcldeck


class MainModule:

    def __init__(self):
        self.cam = module_andorixon.EMCCDCamera()
        self.hacam = module_hamamatsu.HamamatsuCameraMR()
        self.dm = module_deformablemirror.DeformableMirror()
        self.laser = module_laser.CoboltLaser()
        self.daq = module_daq.DAQ()
        # self.tiscam = module_tiscamera.TISCamera()
        self.md = module_mcldeck.MCLMicroDrive()
        # self.thocam = module_thorcamera.UC480Cam()

    def close(self):
        self.cam.shutdown()
        self.hacam.shutdown()
        # self.tiscam.close()
        # self.thocam.close()
        self.laser.all_off()
        self.dm.ResetDM()
        self.daq.Reset_daq()
        self.md.close()
