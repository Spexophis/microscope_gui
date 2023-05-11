from modules import module_andorixon
from modules import module_hamamatsu
from modules import module_tis
from modules import module_deformablemirror
from modules import module_laser
from modules import module_nidaq
from modules import module_mcldeck
from modules import module_mclpiezo


class MainModule:

    def __init__(self):
        self.ccdcam = module_andorixon.EMCCDCamera()
        self.scmoscam = module_hamamatsu.HamamatsuCameraMR()
        self.tiscam = module_tis.TISCamera()
        self.dm = module_deformablemirror.DeformableMirror()
        self.laser = module_laser.CoboltLaser()
        self.daq = module_nidaq.NIDAQ()
        self.md = module_mcldeck.MCLMicroDrive()
        self.pz = module_mclpiezo.MCLNanoDrive()

    def close(self):
        self.ccdcam.close()
        self.scmoscam.close()
        self.tiscam.close()
        self.laser.all_off()
        self.dm.ResetDM()
        self.daq.Reset_daq()
        self.md.close()
        self.pz.close()
