from modules import module_andorixon
from modules import module_hamamatsu
from modules import module_deformablemirror
from modules import module_laser
from modules import module_nidaq
from modules import module_mcldeck
from modules import module_mclpiezo


class MainModule:

    def __init__(self):
        self.ccdcam = module_andorixon.EMCCDCamera()
        self.scmoscam = module_hamamatsu.HamamatsuCameraMR()
        self.dm = module_deformablemirror.DeformableMirror()
        self.laser = module_laser.CoboltLaser()
        self.daq = module_nidaq.NIDAQ()
        self.md = module_mcldeck.MCLMicroDrive()
        self.pz = module_mclpiezo.MCLNanoDrive()

    def close(self):
        self.ccdcam.close()
        self.scmoscam.close()
        self.laser.all_off()
        self.dm.reset_dm()
        self.daq.reset_daq()
        self.md.close()
        self.pz.close()
